# Compatibility and release blockers

Verified 2026-11-13 against FineLAP HF revision `b419aa22947d29907a5567f21b81bf3b39a40449`.

1. **Transformers 5:** the old remote-code loader failed in local Transformers 5.13.0 testing. The extracted local model and fixed RoBERTa config/tokenizer are verified on Transformers 4.57.6. The temporary `<5` constraint stays until real checkpoint parity is measured on 5.x; a successful import alone is insufficient. This document tracks removal.
2. **Rights:** the author GitHub repository has no explicit LICENSE file. The author Hugging Face snapshot used for the six extracted files declares MIT in its model card. This repo credits that precise source and records original file hashes in `source-provenance.md`. Do not copy additional code from the unlicensed GitHub tree or mirror weights; seek written author confirmation before publishing to PyPI.
3. **Compiled torchaudio:** the extracted model still uses torchaudio for load/resampling/Kaldi fbank. It is an optional `model` extra; the base wheel remains importable, but actual inference needs it. A pure replacement with author parity, or an explicit exception, remains a release gate.
4. **Temporal meaning:** the 0.16-second nominal bin comes from 10 ms mel frames and a 16-frame temporal patch. Upstream attention mixes context globally, so this is an index coordinate, not receptive field or onset accuracy. Test with annotated local FX before a public timing claim.
