# Changelog

## Unreleased

- Gate pushes and pull requests with Python 3.10–3.15 offline tests, lint, and a clean wheel-from-sdist import check; keep runtime-specific model parity separate.
- Decode WAV/FLAC with SoundFile so inference no longer requires TorchCodec or shared FFmpeg on newer TorchAudio; preserve TorchAudio resampling and Kaldi fbank, with frontend parity tests.
- Reorganized the README around upstream credit, citation, public GitHub installation, usage, weights, development, license, and support.

## 0.1.0 — 2026-11-13

- Initial inference-only wrapper for pinned FineLAP author checkpoint; public
  wrapper code is MIT, while upstream code and weights are not redistributed.
- Added ready-only resident session, verified checkpoint resolution, JSON CLI and long-file dense-score localization.
- Added nominal time-bin, overlap and threshold-derived event semantics with explicit limits.
- Extracted the six inference-required model files from the MIT-tagged author
  Hugging Face revision; stopped using Transformers remote code at runtime.
- Replaced timm-only layers with local Torch equivalents, pinned RoBERTa
  config/tokenizer and verified exact CPU frame/clip parity on the golden clip.
