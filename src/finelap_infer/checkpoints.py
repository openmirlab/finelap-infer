"""Resolve and verify the pinned HF snapshot for both load and cache_info.

Reads: config/checkpoints.toml, huggingface_hub snapshot_download, SHA-256.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class Checkpoint:
    repo_id: str
    revision: str
    weight_file: str
    sha256: str
    config_sha256: str
    url: str
    source: str
    license: str


@dataclass(frozen=True)
class TextAsset:
    repo_id: str
    revision: str
    files: dict[str, str]


def _config() -> dict:
    return tomllib.loads(files("finelap_infer").joinpath("config/checkpoints.toml").read_text())


def default_checkpoint() -> Checkpoint:
    cfg = _config()
    model = cfg["model"]
    return Checkpoint(**{key: model[key] for key in Checkpoint.__dataclass_fields__})


def default_text_asset() -> TextAsset:
    data = _config()["text"]
    return TextAsset(data["repo_id"], data["revision"], dict(data["files"]))


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


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
            allow_patterns=["config.json", checkpoint.weight_file],
            local_files_only=local_files_only,
        )
    )
    for name in ("config.json", checkpoint.weight_file):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"FineLAP snapshot missing {name}: {directory}")
    if _sha256(directory / checkpoint.weight_file) != checkpoint.sha256:
        raise ValueError(f"FineLAP checkpoint SHA-256 mismatch: {directory / checkpoint.weight_file}")
    if _sha256(directory / "config.json") != checkpoint.config_sha256:
        raise ValueError(f"FineLAP config SHA-256 mismatch: {directory / 'config.json'}")
    return directory


def resolve_text_asset(
    asset: TextAsset, *, cache_dir: str | Path | None = None, local_files_only: bool = False
) -> Path:
    from huggingface_hub import snapshot_download

    directory = Path(snapshot_download(
        asset.repo_id, revision=asset.revision, allow_patterns=list(asset.files),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
    ))
    for name, expected in asset.files.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"FineLAP text asset missing {name}: {directory}")
        if _sha256(path) != expected:
            raise ValueError(f"FineLAP text asset SHA-256 mismatch: {path}")
    return directory
