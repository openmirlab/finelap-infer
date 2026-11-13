# finelap-infer maintainer guide

Public standalone inference-only wrapper for FineLAP. Read README and `docs/compatibility.md` before changes. The upstream author model is loaded from the fixed HF revision in `src/finelap_infer/config/checkpoints.toml`; no upstream code or weights are vendored. The repository's MIT license covers our wrapper only. GitHub FineLAP's missing explicit code LICENSE still blocks vendoring or mirroring upstream code pending clarification; the HF model card's MIT tag does not settle that source-code issue.

## Scope and lifecycle

- Only localization and its native score/time semantics. No training, fine-tuning, datasets, evaluation or experiment commands in the installed surface.
- `checkpoints.py` owns one resolver shared by load and cache inspection. `session.py` owns lifecycle, device, windowing and upstream calls. `results.py` owns output meanings. CLI delegates to the session.
- 10-second windows, 8-second hop, 0.16-second nominal score bins. Overlap uses max. Event intervals are thresholded nominal bins, not sample-accurate boundaries or calibrated probabilities.
- Use local fixed-revision source for tests. Any change to model code/dependencies, windowing, score interpretation or timing must refresh source evidence and upstream parity.
- The source repository is public. No PyPI release, upstream code or weight mirror, or Phonon integration is part of this version.

## Verification

```bash
uv sync --extra model --extra dev
uv run --extra model --extra dev pytest -q
uv run --extra model --extra dev pytest -q -m model
uv run --extra dev ruff check src tests
uv run --extra dev python -m build
```

Before publishing to PyPI: obtain rights clarification, remove or justify the Transformers 5 upper bound with a tracked issue, validate Python 3.10+ matrix, test a clean wheel from sdist and published-name install, verify long-file temporal localization against real annotated riser/downlifter/vinyl, and add a compatible pure default path if the optional torchaudio implementation remains the only runnable backend. README, this file and CHANGELOG change together for public behavior.
