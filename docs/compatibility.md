# Compatibility and release blockers

Verified 2026-11-13 against FineLAP HF revision `b419aa22947d29907a5567f21b81bf3b39a40449`.

1. **Transformers 5:** the upstream remote model did not load in the local Transformers 5.13.0 environment. Transformers 4.57.6 loaded after installing timm 1.0.31. The temporary `<5` constraint is reviewed when upstream updates its HF implementation; remove it only after real checkpoint parity, not an import smoke check. Tracking item: this document; open a repository issue once a private remote exists.
2. **Rights:** the author GitHub repository has no explicit LICENSE file, though the HF model card says MIT. The public wrapper repository contains only original openmirlab code under MIT. Do not copy upstream source or mirror weights, and obtain a written code-license answer before distributing the upstream implementation or publishing a package that fetches it at runtime.
3. **Compiled torchaudio:** upstream's model code imports torchaudio for load/resampling/Kaldi fbank. It is an optional `model` extra here so the base wheel remains importable, but the only real model backend still needs it. OpenMIRLab's no-compiled-core ideal is not fully achieved for inference; a parity-proven pure replacement or explicit org exception is required before a PyPI release.
4. **Temporal meaning:** the 0.16-second nominal bin comes from 10 ms mel frames and a 16-frame temporal patch. Upstream attention mixes context globally, so this is an index coordinate, not receptive field or onset accuracy. Test with annotated local FX before a public timing claim.
