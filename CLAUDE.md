# finelap-infer maintainer guide

Public standalone inference-only package for FineLAP. Original research is by Xiquan Li, Xuenan Xu, Ziyang Ma, Wenxi Chen, Haolin He, Qiuqiang Kong, and Xie Chen; the checkpoint is hosted at `AndreasXi/FineLAP`. Read README, `docs/compatibility.md` and `docs/source-provenance.md` before changes. The inference-required author Python files were extracted from the MIT-tagged Hugging Face snapshot in `src/finelap_infer/config/checkpoints.toml` into `_impl/`. The separate GitHub research tree has no explicit LICENSE and is not the code source. Weights are never bundled. Do not add upstream training/evaluation code or mirror weights. Confirm rights with the author before a PyPI release.

## Scope and lifecycle

- Only localization and its native score/time semantics. No training, fine-tuning, datasets, evaluation or experiment commands in the installed surface.
- `checkpoints.py` owns the two verified asset resolvers shared by load and cache inspection. `session.py` owns lifecycle, device, windowing and calls into the local `_impl` model. `results.py` owns output meanings. CLI delegates to the session.
- `_impl/modeling_finelap.py` decodes WAV/FLAC with SoundFile, then retains TorchAudio resampling and Kaldi fbank. The `model` extra still provides TorchAudio; TorchCodec and shared FFmpeg are not needed for these inputs. `tests/test_audio_decoding.py` checks original PCM16 decode semantics and the public `infer()` path while forcing `torchaudio.load()` to fail.
- 10-second windows, 8-second hop, 0.16-second nominal score bins. Overlap uses max. Event intervals are thresholded nominal bins, not sample-accurate boundaries or calibrated probabilities.
- Use local fixed-revision source for tests. Any change to model code/dependencies, windowing, score interpretation or timing must refresh source evidence and upstream parity.
- The source repository is public; install from GitHub with `pip install 'finelap-infer[model] @ git+https://github.com/openmirlab/finelap-infer.git'`. No PyPI release, weight mirror, or Phonon integration is part of this version. Load-bearing source files carry a title/rationale/`Reads:` navigation header; keep it accurate when editing.

## Verification

```bash
uv sync --extra model --extra dev
uv run --extra model --extra dev pytest -q
uv run --extra model --extra dev pytest -q -m model
uv run --extra dev ruff check src tests
uv run --extra dev python -m build
```

Before publishing to PyPI: obtain rights clarification, remove or justify the Transformers 5 upper bound with a tracked issue, validate Python 3.10+ matrix, test a clean wheel from sdist and published-name install, verify long-file temporal localization against real annotated riser/downlifter/vinyl, and add a compatible pure default path if the optional torchaudio implementation remains the only runnable backend. README, this file and CHANGELOG change together for public behavior.
