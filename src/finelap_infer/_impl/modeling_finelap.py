"""FineLAP model and audio frontend adapted from the pinned author snapshot.

SoundFile reads supported files without TorchCodec; TorchAudio retains the
author's numerical feature pipeline.
Reads: SoundFile for WAV/FLAC decoding; TorchAudio for resampling and Kaldi fbank.
"""

import os

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from transformers import PreTrainedModel, RobertaConfig, RobertaModel, RobertaTokenizer

from .configuration_finelap import FineLAPConfig
from .modeling_eat import EATModel


class FineLAPModel(PreTrainedModel):
    config_class = FineLAPConfig

    def __init__(self, config: FineLAPConfig):
        super().__init__(config)
        self.config = config
        self.audio_encoder = EATModel(config.audio_config)
        self.audio_width = getattr(config.audio_config, 'hidden_size', 768)

        # The FineLAP checkpoint contains the complete RoBERTa encoder. Build
        # its pinned architecture without fetching a second weight file.
        self.text_encoder = RobertaModel(
            RobertaConfig.from_pretrained(config.text_encoder_name), add_pooling_layer=False
        )
        self.text_width = self.text_encoder.config.hidden_size
        self.tokenizer = RobertaTokenizer.from_pretrained(config.text_encoder_name)

        self.embed_size = config.embed_size

        for param in ['temp_global', 'b_global', 'temp_local', 'b_local']:
            val = getattr(config, param, None)
            if val is not None:
                self.register_parameter(param, nn.Parameter(torch.ones([]) * val))

        self.global_audio_proj = nn.Sequential(nn.Linear(self.audio_width, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self.global_text_proj = nn.Sequential(nn.Linear(self.text_width, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))

        self.local_audio_proj_type = config.local_audio_proj_type
        if self.local_audio_proj_type == "rnn":
            self.local_audio_proj = nn.GRU(input_size=self.audio_width, hidden_size=int(self.embed_size / 2), num_layers=2, batch_first=True, bidirectional=True)
        elif self.local_audio_proj_type == "transformer":
            l = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=8, dim_feedforward=self.embed_size * 4, dropout=0.1, activation='relu', batch_first=True)
            self.local_audio_proj = nn.Sequential(nn.Linear(self.audio_width, self.embed_size), nn.TransformerEncoder(l, num_layers=2))
        elif self.local_audio_proj_type == "linear":
            self.local_audio_proj = nn.Sequential(nn.Linear(self.audio_width, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))

        self.post_init()

    def load_audio(self, audio_path, device=None):
        device = device or self.device

        if isinstance(audio_path, (str, os.PathLike)):
            audio_paths = [str(audio_path)]
        else:
            audio_paths = [str(p) for p in audio_path]  # list/tuple/iterable of paths

        mels = []
        target_len = 1024

        for source_path in audio_paths:
            samples, sr = sf.read(source_path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(samples.T.copy())
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.squeeze(0)
            wav = wav - wav.mean()

            mel = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )  # [T, 128]

            if mel.shape[0] < target_len:
                mel = F.pad(mel, (0, 0, 0, target_len - mel.shape[0]))
            else:
                mel = mel[:target_len, :]

            mel = (mel - (-4.268)) / (4.569 * 2)  # [T, 128]
            mels.append(mel)

        mel_batch = torch.stack(mels, dim=0).unsqueeze(1).to(device)  # [B, 1, 1024, 128]
        return mel_batch

    def encode_audio(self, audio_path):
        audio_mel = self.load_audio(audio_path)
        outputs = self.audio_encoder.extract_features(audio_mel)
        raw = outputs['x'] if isinstance(outputs, dict) else outputs
        B, T, D = raw[:, 1:, :].shape
        ds = 8
        patches = raw[:, 1:, :].reshape(B, T // ds, ds, D).mean(dim=2)
        return torch.cat([raw[:, 0:1, :], patches], dim=1)

    def get_global_text_embeds(self, text_labels, device=None):
        device = device or self.device
        t_in = self.tokenizer(text_labels, padding=True, truncation=True, return_tensors="pt").to(device)
        feat = self.text_encoder(input_ids=t_in["input_ids"], attention_mask=t_in["attention_mask"]).last_hidden_state
        return F.normalize(self.global_text_proj(feat[:, 0, :]), dim=-1)

    def get_global_audio_embeds(self, audio_path):
        audio_feats = self.encode_audio(audio_path)
        if self.config.unify_audio_proj:
            audio_embeds = self.local_audio_proj(audio_feats)
            if self.config.local_audio_proj_type == "rnn":
                audio_embeds = audio_embeds[0]
            return F.normalize(audio_embeds[:, 0, :], dim=-1)
        else:
            audio_cls_feat = audio_feats[:, 0, :]
            return F.normalize(self.global_audio_proj(audio_cls_feat), dim=-1)

    def get_dense_audio_embeds(self, audio_path):
        patches = self.encode_audio(audio_path)[:, 1:, :]
        out = self.local_audio_proj(patches)
        embeds = out[0] if self.local_audio_proj_type == "rnn" else out
        return F.normalize(embeds, dim=-1) if self.config.normalize_dense_audio_embeds else embeds

    @torch.no_grad()
    def get_frame_level_score(self, audio_path, text_labels, device=None):
        device = device or self.device
        self.to(device)
        self.eval()

        dense_audio = self.get_dense_audio_embeds(audio_path)  # (B, T, D)
        text_embeds = self.get_global_text_embeds(text_labels, device).unsqueeze(0)  # (1, N, D)

        sim = torch.matmul(text_embeds, dense_audio.transpose(-1, -2))
        if hasattr(self, "temp_local"):
            sim = sim / self.temp_local
        if hasattr(self, "b_local"):
            sim = sim + self.b_local
        return F.sigmoid(sim)

    @torch.no_grad()
    def get_clip_level_score(self, audio_path, text_labels, device=None):
        device = device or self.device
        self.to(device)
        self.eval()

        global_audio = self.get_global_audio_embeds(audio_path)
        global_text = self.get_global_text_embeds(text_labels, device)

        logits = torch.matmul(global_text, global_audio.transpose(-1, -2))
        # return logits
        if hasattr(self, "temp_global"):
            logits = logits / self.temp_global
        if hasattr(self, "b_global"):
            logits = logits + self.b_global
        return F.sigmoid(logits).squeeze(-1)

    def forward(self, audio_path=None, text_labels=None):
        res = {}
        if audio_path is not None:
            res["global_audio_embeds"] = self.get_global_audio_embeds(audio_path) if not self.config.unify_audio_proj else None
            res["dense_audio_embeds"] = self.get_dense_audio_embeds(audio_path)
        if text_labels is not None:
            res["global_text_embeds"] = self.get_global_text_embeds(text_labels)
        return res
