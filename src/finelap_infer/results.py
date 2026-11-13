"""Native FineLAP score and time-coordinate declarations.

Reads: pinned upstream config and FineLAPModel.encode_audio/get_frame_level_score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FrameScore:
    start: float
    end: float
    score: float
    phrase: str


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    peak_score: float
    phrase: str


@dataclass(frozen=True)
class LocalizationResult:
    duration: float
    phrases: tuple[str, ...]
    frames: tuple[FrameScore, ...]
    events: tuple[Event, ...]
    window_starts: tuple[float, ...]
    threshold: float | None
    checkpoint_revision: str
    semantics_version: str = "finelap-localization-v1"
    scope: str = "input_file"
    frame_anchor: str = "nominal_patch_start"
    frame_interval: str = "left_closed_right_open"
    frame_hop_seconds: float = 0.16
    window_seconds: float = 10.0
    window_hop_seconds: float = 8.0
    score_meaning: str = "FineLAP sigmoid similarity; not a calibrated event probability"
    time_meaning: str = "Nominal [start,end) time bins on input-file clock; not exact acoustic onset"
    overlap_aggregation: str = "max score for aligned bins across windows"

    def to_dict(self) -> dict:
        return asdict(self)
