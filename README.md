# finelap-infer

Private, standalone, inference-only wrapper for [FineLAP](https://github.com/xiquan-li/FineLAP). It maps FineLAP's dense audio–text scores onto the input file clock and processes files longer than the upstream model's approximately 10-second frontend limit. The model and its research belong to the FineLAP authors.

**Status:** local private package. The upstream GitHub repository has no explicit code LICENSE as checked 2026-11-13. Its Hugging Face model card says MIT, but that does not by itself grant a license for the GitHub code. This package does not vendor upstream code or checkpoints. Do not publish this package or mirror the checkpoint until the rights are clarified. The pinned HF snapshot executes upstream model code via `trust_remote_code=True` at load time.

## Install

Python 3.10+; the tested environment is Python 3.12, Torch 2.7.0+cu126 CPU mode, Transformers 4.57.6, timm 1.0.31. Install from this checkout:

```bash
uv sync --extra model --extra dev
```

The `model` extra supplies torchaudio, required by the author's HF implementation. The base installation stays importable without torchaudio; `load()` gives an install hint if it is absent. Transformers 5 is temporarily excluded because the pinned upstream remote code failed to load in local testing; see [compatibility tracking](docs/compatibility.md).

## Use

```python
from finelap_infer import FineLAPSession

with FineLAPSession(device="cpu") as model:
    result = model.infer("long_recording.wav", ["A siren sounds."], threshold=0.5)
    print(result.events)  # threshold-derived, approximate spans on input-file clock
    print(result.frames)  # nominal [start, end) bins with raw FineLAP scores
```

Or use `uv run --extra model finelap-infer long_recording.wav 'A siren sounds.' --threshold 0.5`. The CLI prints JSON. `localize(path, phrases, threshold=...)` creates and closes a new session; use `FineLAPSession` for repeated calls. A session supports `load()`, ready-only `infer(...)`, `release()`, `close()`, `status`, `cache_info()`, and a context manager. Sessions are not safe for simultaneous `infer` calls.

Supported input is a path readable by `soundfile`, principally WAV or FLAC. Upstream receives the original path for a file at most 10 seconds. For a longer file, this package reads successive 10-second source-rate windows every 8 seconds, writes float WAV windows into a temporary directory, and asks the pinned model for its 64 dense positions. It removes padded positions beyond real audio, maps each nominal 0.16-second bin to original-file seconds, and takes the maximum score where two windows cover the same bin. Temp files are removed after the call. Memory use is one source window at a time, plus the returned score tracks. The window map is part of the result as `window_starts`.

The `threshold` is caller-selected. If it is omitted, no events are produced, only frame scores. Adjacent bins at or above the threshold become `[start,end)` event intervals. The scores are **sigmoid similarities, not calibrated probabilities**. The interval endpoints are **thresholded model-bin bounds, not exact acoustic onset or offset estimates**. The 0.16-second grid follows the pinned upstream frontend (`10 ms fbank × 16-frame temporal patch`); model attention can use context outside each bin. The result declares `semantics_version="finelap-localization-v1"`, nominal frame anchor, half-open interval policy, window geometry and checkpoint revision. Meaning is tested separately from event-detection accuracy. For shorter than 0.16-second events, different phrase wording, low-level FX, or boundaries, inspect the score track and original audio.

Long-file clip scores and global embeddings are intentionally not exposed: the pinned model computes those for one truncated frontend, and aggregating them across windows would need a separate defined policy. This package covers event localization, not a general-purpose full-file embedding.

## Checkpoint and cache

The default author HF revision and model SHA-256 are in [`config/checkpoints.toml`](src/finelap_infer/config/checkpoints.toml). The author snapshot is downloaded on first `load()`; the large weight file is SHA-256 verified. Hugging Face caches it under its default cache, normally `~/.cache/huggingface/hub`. Supply `cache_dir=` to change the model snapshot cache, or set `HF_HOME` before starting Python for the author's RoBERTa dependency too. `cache_info()` uses the same snapshot resolver as `load()` and never downloads. `device` supports `auto`, `cpu`, `cuda`, and `cuda:N`; `mps` is rejected.

For manual/offline placement, prefill the chosen Hugging Face cache while online: `hf download AndreasXi/FineLAP --revision b419aa22947d29907a5567f21b81bf3b39a40449 --cache-dir /your/cache`. Also cache `roberta-base` with the same tool; the author model constructs that encoder by name and will otherwise try its default cache. Then pass `cache_dir="/your/cache"` and `local_files_only=True` when loading. Use `HF_HOME` before Python starts if you need all dependencies in one offline cache. A `Checkpoint` object allows generic metadata overrides for another verified snapshot; all files remain subject to its SHA-256. Directly passing a loose local folder is not currently supported because Transformers 4.57's local-directory loader fails on this model's transitive relative imports.

## Scope and verification

The shipped wheel contains inference wrapper code, checkpoint metadata, and result types. It contains no training loops, fine-tuning, dataset loaders, evaluation harness, research CLI, checkpoint weights, or copied upstream implementation. Tests include an upstream short-file parity fixture, synthetic boundary/overlap checks and a real-checkpoint late-event smoke test; model-marked tests require the cached checkpoint and optional model dependency. Run:

```bash
uv run --extra model --extra dev pytest -q
uv run --extra model --extra dev pytest -q -m model
uv run --extra dev ruff check src tests
uv run --extra dev python -m build
```

This is not integrated into Phonon. The result schema is package-owned and JSON-serializable; any future adapter must preserve the checkpoint revision, window starts, threshold, score meaning and nominal time-bin meaning.
