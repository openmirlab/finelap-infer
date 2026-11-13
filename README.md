# finelap-infer

Locate sounds described by text in short or long audio with the FineLAP model.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## Why this exists

[FineLAP](https://github.com/xiquan-li/FineLAP) provides frame-level audio–text scores, but its quick-start model loads custom code from Hugging Face and its frontend handles about 10 seconds at a time. This standalone package carries only the inference model code needed for localization, verifies pinned assets, and maps scores from overlapping windows back to the input-file clock. The model and research belong to the FineLAP authors.

**Status:** public inference-only source repository; no PyPI release. The six model implementation files needed for inference are adapted from the fixed author [Hugging Face snapshot](https://huggingface.co/AndreasXi/FineLAP), whose model card declares MIT. The separate [GitHub research repo](https://github.com/xiquan-li/FineLAP) has no explicit LICENSE as checked 2026-11-13; we do not copy from that tree. We do not bundle or mirror checkpoint weights. See [source provenance](docs/source-provenance.md) and [NOTICE](NOTICE).

## Acknowledgments

- Original research and model: Xiquan Li, Xuenan Xu, Ziyang Ma, Wenxi Chen, Haolin He, Qiuqiang Kong, and Xie Chen ([FineLAP paper](https://arxiv.org/abs/2604.01155)).
- Original research source: [xiquan-li/FineLAP](https://github.com/xiquan-li/FineLAP); adapted inference code and checkpoint source: [AndreasXi/FineLAP](https://huggingface.co/AndreasXi/FineLAP).
- Checkpoint host: [Hugging Face, AndreasXi/FineLAP](https://huggingface.co/AndreasXi/FineLAP). OpenMIRLab did not train this model.

## Citation

Please cite the [FineLAP paper](https://arxiv.org/abs/2604.01155) as requested by its authors:

```bibtex
@article{li2026finelap,
  title={FineLAP: Taming Heterogeneous Supervision for Fine-grained Language-Audio Pretraining},
  author={Li, Xiquan and Xu, Xuenan and Ma, Ziyang and Chen, Wenxi and He, Haolin and Kong, Qiuqiang and Chen, Xie},
  journal={Proc. ACL},
  year={2026}
}
```

## Features

- Dense phrase scores on nominal 0.16-second bins and optional thresholded event spans.
- Overlapping 10-second windows for longer WAV/FLAC files, mapped to input-file seconds.
- Verified checkpoint loading, cache inspection, explicit CPU/CUDA device selection, and CLI JSON output.

## Scope

This package supports localization only. It does not expose full-file clip scores or global embeddings, and it does not train, fine-tune, or evaluate the model.

## Install

Python 3.10+; the tested model environment is Python 3.12, Torch 2.7.0+cu126 CPU mode and Transformers 4.57.6. Install from public GitHub:

```bash
pip install 'finelap-infer[model] @ git+https://github.com/openmirlab/finelap-infer.git'
```

There is no PyPI release. For a local development checkout, run `uv sync --extra model --extra dev`. The `model` extra supplies torchaudio for the author's Kaldi fbank preprocessing; its binary must match Torch. The base installation stays importable without torchaudio; `load()` gives an install hint if it is absent. The inference model code is local and does not use `trust_remote_code`. Transformers 5 is temporarily excluded pending parity testing; see [compatibility tracking](docs/compatibility.md).

## Quick start

```python
from finelap_infer import FineLAPSession

with FineLAPSession(device="cpu") as model:
    result = model.infer("long_recording.wav", ["A siren sounds."], threshold=0.5)
    print(result.events)  # threshold-derived, approximate spans on input-file clock
    print(result.frames)  # nominal [start, end) bins with raw FineLAP scores
```

Or use `finelap-infer long_recording.wav 'A siren sounds.' --threshold 0.5`. The CLI prints JSON. `localize(path, phrases, threshold=...)` creates and closes a new session; use `FineLAPSession` for repeated calls. A session supports `load()`, ready-only `infer(...)`, `release()`, `close()`, `status`, `cache_info()`, and a context manager. Sessions are not safe for simultaneous `infer` calls.

## Usage and output

Supported input is a path readable by `soundfile`, principally WAV or FLAC. Upstream receives the original path for a file at most 10 seconds. For a longer file, this package reads successive 10-second source-rate windows every 8 seconds, writes float WAV windows into a temporary directory, and asks the pinned model for its 64 dense positions. It removes padded positions beyond real audio, maps each nominal 0.16-second bin to original-file seconds, and takes the maximum score where two windows cover the same bin. Temp files are removed after the call. Memory use is one source window at a time, plus the returned score tracks. The window map is part of the result as `window_starts`.

The `threshold` is caller-selected. If it is omitted, no events are produced, only frame scores. Adjacent bins at or above the threshold become `[start,end)` event intervals. The scores are **sigmoid similarities, not calibrated probabilities**. The interval endpoints are **thresholded model-bin bounds, not exact acoustic onset or offset estimates**. The 0.16-second grid follows the pinned upstream frontend (`10 ms fbank × 16-frame temporal patch`); model attention can use context outside each bin. The result declares `semantics_version="finelap-localization-v1"`, nominal frame anchor, half-open interval policy, window geometry and checkpoint revision. Meaning is tested separately from event-detection accuracy. For shorter than 0.16-second events, different phrase wording, low-level FX, or boundaries, inspect the score track and original audio.

Long-file clip scores and global embeddings are intentionally not exposed: the pinned model computes those for one truncated frontend, and aggregating them across windows would need a separate defined policy. This package covers event localization, not a general-purpose full-file embedding.

## Checkpoint and cache

The pinned author weight/config and RoBERTa tokenizer/config revisions and SHA-256 values are in [`config/checkpoints.toml`](src/finelap_infer/config/checkpoints.toml). The two verified snapshots are downloaded on first `load()` into the Hugging Face cache, normally `~/.cache/huggingface/hub`; pass `cache_dir=` to change it. `cache_info()` checks the same two resolvers without downloading. `device` supports `auto`, `cpu`, `cuda`, and `cuda:N`; `mps` is rejected.

For manual/offline placement, prefill the chosen Hugging Face cache while online:

```bash
hf download AndreasXi/FineLAP config.json model.safetensors \
  --revision b419aa22947d29907a5567f21b81bf3b39a40449 --cache-dir /your/cache
hf download roberta-base config.json vocab.json merges.txt tokenizer.json tokenizer_config.json \
  --revision e2da8e2f811d1448a5b465c236feacd80ffbac7b --cache-dir /your/cache
```

Then pass `cache_dir="/your/cache"` and `local_files_only=True` when loading. All files are SHA-256 checked. The FineLAP checkpoint includes RoBERTa encoder weights, so only RoBERTa config/tokenizer files are fetched. `Checkpoint` and `TextAsset` allow verified asset overrides.

## What this project will never bundle

The wheel contains the six adapted inference model files, localization code, checkpoint metadata, and result types. It never bundles checkpoint weights, training loops, fine-tuning, dataset loaders, evaluation harnesses, or the research CLI.

## Development

Tests include an upstream short-file parity fixture, synthetic boundary/overlap checks and a real-checkpoint late-event smoke test; model-marked tests require the cached checkpoint and optional model dependency. From a checkout, run:

```bash
uv run --extra model --extra dev pytest -q
uv run --extra model --extra dev pytest -q -m model
uv run --extra dev ruff check src tests
uv run --extra dev python -m build
```

This is not integrated into Phonon. The result schema is package-owned and JSON-serializable; any future adapter must preserve the checkpoint revision, window starts, threshold, score meaning and nominal time-bin meaning.

## License

Package source is MIT licensed. The adapted model code comes from the MIT-tagged author Hugging Face snapshot; the separate research GitHub tree has no explicit license. Checkpoint rights are documented separately in the [manifest](src/finelap_infer/config/checkpoints.toml). See [LICENSE](LICENSE) and [NOTICE](NOTICE). Confirm source rights with the author before a PyPI release.

## Support

Report package bugs in [openmirlab/finelap-infer issues](https://github.com/openmirlab/finelap-infer/issues). For research or model questions, see the [original FineLAP repository](https://github.com/xiquan-li/FineLAP).
