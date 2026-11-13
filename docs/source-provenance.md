# FineLAP inference source provenance

Source: `AndreasXi/FineLAP` on Hugging Face at revision
`b419aa22947d29907a5567f21b81bf3b39a40449`. The model card declares
MIT. The separate GitHub research repository lacks a LICENSE; no files were
copied from that checkout. The package includes these six source files because
they define the checkpoint architecture and frame-score computation:

| File in `_impl/` | SHA-256 of original HF file |
| --- | --- |
| `configuration_eat.py` | `761f19555f0d53aa7725ffb83a6e4eec652af97244134ea5147edca71be9b9e5` |
| `configuration_finelap.py` | `89343bfc8cf63a5d2ddd8925c5d2b6ceb7e1a62ff76e05f4b01901aac0400a11` |
| `eat_model.py` | `ce1d10ab3408d1a13457781dac6ef37f3d43bb40df7f0b51ce8d6358b181ecab` |
| `eat_model_core.py` | `f0157ab1d7f59e07db6da9d7845099dac7b818e1ea69322936da83364d8a2768` |
| `modeling_eat.py` | `41eaf4b87923a56ef13899cbbe6d6ec9642d1f58db2e0b2661aae611eccfce9f` |
| `modeling_finelap.py` | `742036dbb2987a9815d3ca09a3a16774de54b8cab6ed1b5f03baca3719baf6e3` |

Changes: removed the plot helper, replaced timm's required layer functions
with `layers.py`, built RoBERTa from its pinned config instead of fetching a
second weight file, loaded the author model through the local class, and
replaced `torchaudio.load()` with SoundFile WAV/FLAC decoding while retaining
the original resampling and Kaldi fbank pipeline.
No train scripts, datasets, evaluation metrics, or model weights are shipped.

On two real Rytho sample previews (10-second 16 kHz mono files), the local
model's `2 × 2 × 64` frame scores were bit-identical to the author's HF
remote-code model in the same CPU runtime. Those user audio files are not
committed here.

The committed `tests/fixtures/upstream_cpu.npz` was captured with the
unmodified author model before extraction. Under the recorded CPU environment
(Torch 2.7.0+cu126, Transformers 4.57.6, four threads), the extracted path
returned bit-identical frame and clip scores on that fixture. The package
regression test covers frame and clip scores. On Torch 2.15.0+cu132 and
TorchAudio 2.11.1+cu132, the same fixture differs by at most 4.47e-7 in frame
scores on a 0.918 score range and 1.49e-8 in clip score; the version-guarded
test checks this tolerance. Real long-file localization remains a separate test.
