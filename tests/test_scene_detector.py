"""
tests/test_scene_detector.py
─────────────────────────────
Test kriteri (CLAUDE.md):
  input:  10 dakikalık video
  beklenti: 10-50 arası sahne tespit edilmeli
  kontrol: her sahne için keyframe dosyası var mı?

Gerçek 10 dakikalık ERP videosu yerine abrupt renk değişimli sentetik
video kullanılır (hızlı, deterministik, ortam-bağımsız).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.scene_detector import Scene, detect_scenes

CONFIG = {"threshold": 30.0, "min_scene_len": 5, "keyframe_position": "mid"}


def _make_synthetic_video(path: Path, fps: int = 10, block_frames: int = 20) -> int:
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]  # BGR
    total = 0
    for color in colors:
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        for _ in range(block_frames):
            writer.write(frame)
            total += 1
    writer.release()
    return total


@pytest.fixture
def synthetic_video(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    _make_synthetic_video(video_path)
    return video_path


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    # detect_scenes() output/ altına hardcoded yazıyor — testleri izole et.
    monkeypatch.chdir(tmp_path)


def test_detect_scenes_returns_multiple_scenes(synthetic_video):
    scenes = detect_scenes(synthetic_video, CONFIG)
    assert len(scenes) >= 2
    assert all(isinstance(s, Scene) for s in scenes)


def test_detect_scenes_keyframes_exist_on_disk(synthetic_video):
    scenes = detect_scenes(synthetic_video, CONFIG)
    assert len(scenes) > 0
    for s in scenes:
        assert s.keyframe_path
        kf = Path(s.keyframe_path)
        assert kf.exists() and kf.stat().st_size > 0


def test_detect_scenes_writes_valid_scenes_json(synthetic_video):
    scenes = detect_scenes(synthetic_video, CONFIG)
    scenes_json_path = Path("output") / "scenes.json"
    assert scenes_json_path.exists()
    data = json.loads(scenes_json_path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == len(scenes)
    assert data[0]["scene_id"] == 0 and "keyframe_path" in data[0]


def test_detect_scenes_keyframe_position_variants(synthetic_video):
    base = {"threshold": 30.0, "min_scene_len": 5}
    scenes_start = detect_scenes(synthetic_video, {**base, "keyframe_position": "start"})
    scenes_end = detect_scenes(synthetic_video, {**base, "keyframe_position": "end"})
    assert len(scenes_start) == len(scenes_end)
    multi = [(a, b) for a, b in zip(scenes_start, scenes_end) if a.end_frame > a.start_frame]
    assert len(multi) > 0


def test_detect_scenes_missing_video_raises():
    with pytest.raises(FileNotFoundError):
        detect_scenes("nonexistent_video_xyz.mp4", CONFIG)
