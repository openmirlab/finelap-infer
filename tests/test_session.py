"""Check timing, overlap, threshold semantics, lifecycle and upstream parity."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import transformers

from finelap_infer import FineLAPSession, default_checkpoint, default_text_asset

FIXTURES = Path(__file__).parent / "fixtures"


class FakeFineLAP:
    def get_frame_level_score(self, paths, phrases, device=None):
        arr = np.full((1, len(phrases), 64), 0.1 if "window-0" in paths[0] else 0.2, dtype=np.float32)
        if "window-0" in paths[0]:
            arr[0, 0, 55] = 0.7
        else:
            arr[0, 0, 5] = 0.9  # same original-file bin: 8.8 seconds
        return torch.from_numpy(arr)


def test_window_overlap_and_input_file_clock(tmp_path):
    audio = tmp_path / "long.wav"
    sf.write(audio, np.zeros(18 * 16000, dtype=np.float32), 16000, subtype="FLOAT")
    session = FineLAPSession(device="cpu")
    session._model = FakeFineLAP()
    session._status = "ready"
    result = session.infer(audio, ["a siren"], threshold=0.5)
    assert result.window_starts == (0.0, 8.0)
    assert result.semantics_version == "finelap-localization-v1"
    assert result.frame_anchor == "nominal_patch_start"
    assert len(result.frames) == 113
    assert result.frames[-1].end == 18.0
    peak = [f for f in result.frames if abs(f.start - 8.8) < 1e-6]
    assert len(peak) == 1
    assert peak[0].score == pytest.approx(0.9)
    assert result.events[0].start == pytest.approx(8.8)
    assert result.events[0].end == pytest.approx(8.96)
    assert result.events[0].peak_score == pytest.approx(0.9)
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()


def test_input_and_threshold_guards():
    session = FineLAPSession(device="cpu")
    with pytest.raises(RuntimeError, match="load"):
        session.infer(FIXTURES / "tone_silence.wav", ["tone"])
    session._model = FakeFineLAP()
    session._status = "ready"
    with pytest.raises(ValueError, match="phrases"):
        session.infer(FIXTURES / "tone_silence.wav", [])
    with pytest.raises(ValueError, match="threshold"):
        session.infer(FIXTURES / "tone_silence.wav", ["tone"], threshold=1.1)


def test_result_contract_import_and_config():
    checkpoint = default_checkpoint()
    assert len(checkpoint.revision) == 40
    assert len(checkpoint.sha256) == 64
    assert len(checkpoint.config_sha256) == 64
    assert len(default_text_asset().files) == 5
    assert FineLAPSession(device="cpu").status == "unloaded"


@pytest.mark.model
def test_pinned_upstream_short_audio_parity():
    """Numerical parity to the unmodified author model on true silence and a tone."""
    exact_env = torch.__version__ == "2.7.0+cu126"
    compatible_env = torch.__version__ == "2.15.0+cu132"
    if not (exact_env or compatible_env) or transformers.__version__ != "4.57.6":
        pytest.skip("Fixture verified on Torch 2.7.0+cu126 or 2.15.0+cu132, Transformers 4.57.6 CPU")
    torch.set_num_threads(4)
    expected = np.load(FIXTURES / "upstream_cpu.npz")
    with FineLAPSession(device="cpu", local_files_only=True) as session:
        result = session.infer(FIXTURES / "tone_silence.wav", ["A sine tone sounds."])
        with torch.inference_mode():
            raw_frames = session._model.get_frame_level_score(
                [str(FIXTURES / "tone_silence.wav")], ["A sine tone sounds."], device="cpu"
            ).detach().numpy()
            raw_clip = session._model.get_clip_level_score(
                [str(FIXTURES / "tone_silence.wav")], ["A sine tone sounds."], device="cpu"
            ).detach().numpy()
        assert session.cache_info()["cached"]
    actual = np.array([frame.score for frame in result.frames])
    if exact_env:
        assert np.array_equal(raw_frames, expected["frames"])
        assert np.array_equal(raw_clip, expected["clip"])
    else:
        np.testing.assert_allclose(raw_frames, expected["frames"], rtol=1e-6, atol=5e-7)
        np.testing.assert_allclose(raw_clip, expected["clip"], rtol=1e-6, atol=5e-7)
    np.testing.assert_allclose(actual, expected["frames"][0, 0, :len(actual)], rtol=1e-6, atol=1e-7)
    assert actual.max() > 0.8
    assert actual.min() < 0.01


@pytest.mark.model
def test_real_model_localizes_after_first_window(tmp_path):
    """Real checkpoint must see a source placed past the upstream 10-second limit."""
    if torch.__version__ not in ("2.7.0+cu126", "2.15.0+cu132"):
        pytest.skip("Threshold fixture verified on Torch 2.7.0+cu126 and 2.15.0+cu132 CPU")
    torch.set_num_threads(4)
    rate = 16000
    wave = np.zeros(18 * rate, dtype=np.float32)
    time = np.arange(2 * rate, dtype=np.float32) / rate
    wave[12 * rate : 14 * rate] = 0.16 * np.sin(2 * np.pi * 440 * time)
    path = tmp_path / "late-tone.wav"
    sf.write(path, wave, rate, subtype="FLOAT")
    with FineLAPSession(device="cpu", local_files_only=True) as session:
        result = session.infer(path, ["A sine tone sounds."], threshold=0.5)
    assert result.window_starts == (0.0, 8.0)
    assert result.frames[-1].end == 18.0
    assert any(event.start <= 12.5 and event.end >= 13.5 for event in result.events)
