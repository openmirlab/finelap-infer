"""Keep WAV/FLAC decoding compatible with the original TorchAudio frontend."""

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from finelap_infer import FineLAPSession
from finelap_infer._impl.modeling_finelap import FineLAPModel


@pytest.mark.parametrize("format_name,channels", [("WAV", 1), ("WAV", 2), ("FLAC", 2)])
@pytest.mark.parametrize("sample_rate", [16000, 22050])
def test_frontend_without_torchcodec_matches_original_decode(
    tmp_path, monkeypatch, format_name, channels, sample_rate
):
    # PCM16 has an exact, backend-independent float32 decode: int / 32768.
    # Include real silence, unequal channels and a non-16 kHz resampling case.
    frames = sample_rate // 4
    pcm = np.zeros((frames, channels), dtype=np.int16)
    t = np.arange(frames // 2, dtype=np.float64) / sample_rate
    pcm[frames // 2 :, 0] = np.round(12000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    if channels == 2:
        pcm[frames // 2 :, 1] = np.round(9000 * np.sin(2 * np.pi * 680 * t)).astype(np.int16)
    path = tmp_path / f"input.{format_name.lower()}"
    sf.write(path, pcm, sample_rate, format=format_name, subtype="PCM_16")

    # Preserve the original channel averaging, resampling and Kaldi features.
    original_decode = torch.from_numpy(pcm.astype(np.float32).T / 32768.0)
    wav = original_decode.mean(dim=0, keepdim=True) if channels > 1 else original_decode
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16000)
    wav = wav.squeeze(0)
    wav = wav - wav.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0), htk_compat=True, sample_frequency=16000,
        use_energy=False, window_type="hanning", num_mel_bins=128,
        dither=0.0, frame_shift=10,
    )
    expected = (F.pad(mel, (0, 0, 0, 1024 - mel.shape[0])) - (-4.268)) / (4.569 * 2)

    def missing_torchcodec(*_args, **_kwargs):
        raise ImportError("TorchCodec is required for load_with_torchcodec")

    monkeypatch.setattr(torchaudio, "load", missing_torchcodec)
    actual = FineLAPModel.load_audio(None, path, device="cpu")[0, 0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_infer_reads_wav_and_flac_without_torchcodec(tmp_path, monkeypatch):
    rate = 22050
    pcm = np.zeros(rate, dtype=np.int16)
    t = np.arange(rate // 2, dtype=np.float64) / rate
    pcm[rate // 2 :] = np.round(12000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    paths = [tmp_path / "input.wav", tmp_path / "input.flac"]
    for path in paths:
        sf.write(path, pcm, rate, subtype="PCM_16")

    def missing_torchcodec(*_args, **_kwargs):
        raise ImportError("TorchCodec is required for load_with_torchcodec")

    monkeypatch.setattr(torchaudio, "load", missing_torchcodec)

    class FrontendModel:
        def get_frame_level_score(self, paths, phrases, device=None):
            mel = FineLAPModel.load_audio(None, paths, device=device)
            return mel.new_full((1, len(phrases), 64), mel.abs().mean().item())

    session = FineLAPSession(device="cpu")
    session._model = FrontendModel()
    session._status = "ready"
    wav_result = session.infer(paths[0], ["A sine tone sounds."])
    flac_result = session.infer(paths[1], ["A sine tone sounds."])
    assert wav_result.frames == flac_result.frames
    assert len(wav_result.frames) == 7
    assert wav_result.frames[0].score > 0
