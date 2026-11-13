"""Resident FineLAP inference and long-file window/time mapping.

Reads: checkpoints.resolve_snapshot, upstream HF model's frame/clip APIs,
soundfile for input duration and temporary waveform windows, results dataclasses.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import soundfile as sf

from .checkpoints import Checkpoint, default_checkpoint, resolve_snapshot
from .results import Event, FrameScore, LocalizationResult

FRAME_HOP = 0.16  # 10 ms fbank frames * 16 mel frames per temporal patch.
WINDOW_SECONDS = 10.0  # under upstream's 1024 x 10 ms fbank limit.
WINDOW_HOP = 8.0  # 2 s overlap; both offsets are multiples of FRAME_HOP.


def _device(value: str) -> str:
    import torch

    if value == "mps":
        raise ValueError("FineLAP does not support device='mps'")
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return value
    if value == "cuda" or value.startswith("cuda:"):
        target = torch.device(value)
        if not torch.cuda.is_available() or target.index is not None and target.index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device is unavailable: {value}")
        return value
    raise ValueError(f"Unsupported device: {value}")


def _starts(duration: float) -> list[float]:
    starts = [0.0]
    while starts[-1] + WINDOW_SECONDS < duration - 1e-8:
        starts.append(starts[-1] + WINDOW_HOP)
    return starts


def _events(frames: tuple[FrameScore, ...], phrases: tuple[str, ...], threshold: float | None) -> tuple[Event, ...]:
    if threshold is None:
        return ()
    found = []
    for phrase in phrases:
        selected = sorted((f for f in frames if f.phrase == phrase), key=lambda f: f.start)
        active = None
        for frame in selected:
            if frame.score >= threshold:
                if active is None:
                    active = [frame.start, frame.end, frame.score]
                elif frame.start <= active[1] + 1e-6:
                    active[1] = frame.end
                    active[2] = max(active[2], frame.score)
                else:
                    found.append(Event(active[0], active[1], active[2], phrase))
                    active = [frame.start, frame.end, frame.score]
            elif active is not None:
                found.append(Event(active[0], active[1], active[2], phrase))
                active = None
        if active is not None:
            found.append(Event(active[0], active[1], active[2], phrase))
    return tuple(found)


class FineLAPSession:
    """Independent resident model. Sessions are not safe for concurrent infer calls."""

    def __init__(
        self,
        *,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        checkpoint: Checkpoint | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.device = device
        self.cache_dir = cache_dir
        self.checkpoint = checkpoint or default_checkpoint()
        self.local_files_only = local_files_only
        self._model = None
        self._status = "unloaded"

    @property
    def status(self) -> str:
        return self._status

    def cache_info(self) -> dict:
        try:
            path = resolve_snapshot(
                self.checkpoint,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )
        except (OSError, ValueError, FileNotFoundError):
            return {"cached": False, "revision": self.checkpoint.revision, "path": None}
        return {"cached": True, "revision": self.checkpoint.revision, "path": str(path)}

    def load(self) -> FineLAPSession:
        if self._status == "closed":
            raise RuntimeError("FineLAP session is closed")
        if self._model is not None:
            return self
        resolved_device = _device(self.device)
        resolve_snapshot(
            self.checkpoint,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        try:
            import torchaudio  # noqa: F401 - required by upstream remote model code
        except ImportError as exc:
            raise ImportError("FineLAP needs torchaudio; install finelap-infer[model]") from exc
        from transformers import AutoModel

        # The pinned remote code is executed, but never copied into our wheel.
        # Transformers 4.57's local-directory dynamic-module loader fails to
        # copy FineLAP's transitive relative imports; the pinned repo-ID path
        # correctly resolves them from the same verified snapshot.
        self._model = AutoModel.from_pretrained(
            self.checkpoint.repo_id,
            revision=self.checkpoint.revision,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
        ).to(resolved_device).eval()
        self._status = "ready"
        return self

    def infer(
        self, audio_path: str | Path, phrases: list[str] | tuple[str, ...], *, threshold: float | None = None
    ) -> LocalizationResult:
        """Map dense score bins to the original file clock, with optional thresholded spans."""
        if self._status != "ready":
            raise RuntimeError("Call load() before infer()")
        if not phrases or any(not isinstance(p, str) or not p.strip() for p in phrases):
            raise ValueError("phrases must be nonempty strings")
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]; scores are not calibrated probabilities")
        path = Path(audio_path).expanduser().resolve()
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        if duration <= 0:
            raise ValueError("Audio must be nonempty")
        phrases = tuple(phrases)
        starts = _starts(duration)
        scores: dict[tuple[int, int], float] = {}
        import torch

        with tempfile.TemporaryDirectory(prefix="finelap-infer-") as tmpdir, sf.SoundFile(str(path)) as source:
                for window_number, start in enumerate(starts):
                    # Direct short-file path preserves upstream preprocessing exactly.
                    if len(starts) == 1:
                        window_path = path
                    else:
                        source.seek(round(start * info.samplerate))
                        samples = source.read(round(WINDOW_SECONDS * info.samplerate), dtype="float32", always_2d=True)
                        window_path = Path(tmpdir) / f"window-{window_number}.wav"
                        sf.write(str(window_path), samples, info.samplerate, subtype="FLOAT")
                    with torch.inference_mode():
                        array = self._model.get_frame_level_score(
                            [str(window_path)], list(phrases), device=_device(self.device)
                        ).detach().cpu().numpy()[0]
                    if array.shape != (len(phrases), 64):
                        raise RuntimeError(f"Unexpected FineLAP frame shape: {array.shape}")
                    for phrase_index in range(len(phrases)):
                        for local_index in range(64):
                            absolute_start = start + local_index * FRAME_HOP
                            if absolute_start >= duration - 1e-9 or absolute_start >= start + WINDOW_SECONDS:
                                break
                            grid_index = round(absolute_start / FRAME_HOP)
                            key = (phrase_index, grid_index)
                            scores[key] = max(scores.get(key, -1.0), float(array[phrase_index, local_index]))
        frames = tuple(
            FrameScore(
                start=round(index * FRAME_HOP, 9),
                end=round(min(duration, (index + 1) * FRAME_HOP), 9),
                score=value,
                phrase=phrases[phrase_index],
            )
            for (phrase_index, index), value in sorted(scores.items())
        )
        return LocalizationResult(
            duration=duration,
            phrases=phrases,
            frames=frames,
            events=_events(frames, phrases, threshold),
            window_starts=tuple(starts),
            threshold=threshold,
            checkpoint_revision=self.checkpoint.revision,
        )

    def release(self) -> None:
        self._model = None
        if self._status != "closed":
            self._status = "unloaded"

    def close(self) -> None:
        self.release()
        self._status = "closed"

    def __enter__(self) -> FineLAPSession:  # noqa: PYI034 - Python 3.10 lacks typing.Self
        return self.load()

    def __exit__(self, *_exc: object) -> None:
        self.close()


def localize(audio_path: str | Path, phrases: list[str] | tuple[str, ...], **kwargs: object) -> LocalizationResult:
    """One-shot convenience path; creates and closes its own model session."""
    threshold = kwargs.pop("threshold", None)
    with FineLAPSession(**kwargs) as session:
        return session.infer(audio_path, phrases, threshold=threshold)
