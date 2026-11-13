"""Resolve and verify the pinned HF snapshot for both load and cache_info.

Reads: config/checkpoints.toml, huggingface_hub snapshot_download, SHA-256.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class Checkpoint:
    repo_id: str
    revision: str
    weight_file: str
    sha256: str
    url: str
    source: str
    license: str


def default_checkpoint() -> Checkpoint:
    cfg = tomllib.loads(files("finelap_infer").joinpath("config/checkpoints.toml").read_text())
    model = cfg["model"]
    return Checkpoint(**{key: model[key] for key in Checkpoint.__dataclass_fields__})


def resolve_snapshot(
    checkpoint: Checkpoint,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return a verified snapshot from the configurable HF cache."""
    from huggingface_hub import snapshot_download

    directory = Path(
        snapshot_download(
            checkpoint.repo_id,
            revision=checkpoint.revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            allow_patterns=["*.py", "*.json", "*.safetensors"],
            local_files_only=local_files_only,
        )
    )
    for name in ("config.json", "modeling_finelap.py", checkpoint.weight_file):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"FineLAP snapshot missing {name}: {directory}")
    with (directory / checkpoint.weight_file).open("rb") as weight_stream:
        digest = hashlib.file_digest(weight_stream, "sha256").hexdigest()
    if digest != checkpoint.sha256:
        raise ValueError(f"FineLAP checkpoint SHA-256 mismatch: {directory / checkpoint.weight_file}")
    return directory
