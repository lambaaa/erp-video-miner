"""
pipeline/scene_detector.py
───────────────────────────
MODÜL 1 — PySceneDetect ile sahne değişimlerini tespit et, her sahnenin
temsil frame'ini çıkar.

Kurulum:
    pip install scenedetect[opencv] rich
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

try:
    from scenedetect import ContentDetector, SceneManager, open_video
except ImportError:
    raise ImportError(
        "PySceneDetect kurulu değil.\n"
        "  pip install scenedetect[opencv]"
    )

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

OUTPUT_DIR    = Path("output")
KEYFRAMES_DIR = OUTPUT_DIR / "keyframes"
SCENES_JSON   = OUTPUT_DIR / "scenes.json"


@dataclass
class Scene:
    scene_id:      int
    start_frame:   int
    end_frame:     int
    start_time:    float   # saniye
    end_time:      float   # saniye
    duration:      float   # saniye
    keyframe_path: str     # output/keyframes/scene_0042.png


def _select_keyframe_index(start_frame: int, end_frame: int, position: str) -> int:
    if position == "start":
        return start_frame
    if position == "end":
        return end_frame
    return (start_frame + end_frame) // 2


def _extract_and_save_keyframe(cap: cv2.VideoCapture, frame_idx: int, out_path: Path) -> bool:
    # Not: CAP_PROP_POS_FRAMES sıkıştırılmış kodeklerde (H.264 vb.) frame-exact
    # değildir, en yakın GOP sınırına yuvarlanabilir. POC için kabul edilebilir.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), frame))


def _write_scenes_json(scenes: list[Scene], out_path: Path = SCENES_JSON) -> None:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in scenes], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[SCENE HATA] scenes.json yazılamadı: {e}")


def detect_scenes(video_path: str | Path, config: dict) -> list[Scene]:
    """
    Girdi:  video_path, config (config.yaml'dan scene_detection bölümü)
    Çıktı:  list[Scene]
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    threshold         = config.get("threshold", 30.0)
    min_scene_len     = config.get("min_scene_len", 15)
    keyframe_position = config.get("keyframe_position", "mid")

    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
        )
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
    except Exception as e:
        print(f"[SCENE HATA] Sahne tespiti başarısız: {e}")
        return []

    if not scene_list:
        # Tüm video tek sahne — downstream modüllerin en az 1 frame alabilmesi için
        # videonun tamamını kapsayan tek bir sahne üret.
        total_frames = int(video.duration.frame_num)
        scene_list = [(video.base_timecode, video.base_timecode + total_frames)]

    cap = cv2.VideoCapture(str(video_path))
    results: list[Scene] = []

    with Progress(
        TextColumn("[SCENE] Keyframe çıkarımı"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("keyframes", total=len(scene_list))
        for scene_id, (start_tc, end_tc) in enumerate(scene_list):
            try:
                start_frame = start_tc.frame_num
                end_frame   = max(end_tc.frame_num - 1, start_frame)
                start_time  = start_tc.seconds
                end_time    = end_tc.seconds

                target_idx = _select_keyframe_index(start_frame, end_frame, keyframe_position)
                keyframe_path = KEYFRAMES_DIR / f"scene_{scene_id:04d}.png"
                ok = _extract_and_save_keyframe(cap, target_idx, keyframe_path)
                kf_str = str(keyframe_path) if ok else ""
                if not ok:
                    print(f"[SCENE HATA] scene_{scene_id:04d}: keyframe çıkarılamadı")

                results.append(Scene(
                    scene_id=scene_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    keyframe_path=kf_str,
                ))
            except Exception as e:
                print(f"[SCENE HATA] scene_{scene_id:04d} işlenemedi: {e}")
            finally:
                progress.update(task, advance=1)

    cap.release()
    _write_scenes_json(results)
    return results


if __name__ == "__main__":
    import sys

    import yaml

    if len(sys.argv) < 2:
        print("Kullanım: python -m pipeline.scene_detector video.mp4")
        sys.exit(1)

    cfg = {"threshold": 30.0, "min_scene_len": 15, "keyframe_position": "mid"}
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("scene_detection", cfg)
    except FileNotFoundError:
        pass

    scenes = detect_scenes(sys.argv[1], cfg)
    print(f"\n[SCENE] Toplam {len(scenes)} sahne tespit edildi.")
    for s in scenes[:10]:
        print(
            f"  scene_{s.scene_id:04d} | {s.start_time:6.1f}s-{s.end_time:6.1f}s "
            f"({s.duration:5.1f}s) -> {s.keyframe_path}"
        )
