"""
pipeline/batch_enricher.py
────────────────────────────
ADIM 1 (event_builder.py öncesi hazırlık) — Tüm sahnelerin OCR + CLIP +
transkript sonuçlarını üret, EnrichedScene'e birleştir ve cache'le.

Girdi:  output/scenes.json, output/transcript_aligned.json
Çıktı:  output/enriched_scenes.json
"""

import json
from dataclasses import asdict
from pathlib import Path

import yaml
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from pipeline.clip_classifier import ERPScreenClassifier
from pipeline.frame_ocr import TurkishERPOCR, _avg_confidence
from pipeline.models import EnrichedScene
from pipeline.scene_detector import Scene
from pipeline.transcript_parser import SceneTranscript

OUTPUT_DIR       = Path("output")
SCENES_JSON      = OUTPUT_DIR / "scenes.json"
TRANSCRIPT_JSON  = OUTPUT_DIR / "transcript_aligned.json"
ENRICHED_JSON    = OUTPUT_DIR / "enriched_scenes.json"

LIST_TABLE_REGION_THRESHOLD = 150
MENU_REGION_THRESHOLD       = 8
OCR_TITLE_CONFIDENCE_FLOOR  = 0.65
CLIP_CONFIDENCE_FLOOR       = 0.55


def _load_scenes() -> list[Scene]:
    with open(SCENES_JSON, "r", encoding="utf-8") as f:
        return [Scene(**d) for d in json.load(f)]


def _load_transcripts() -> dict[int, SceneTranscript]:
    if not TRANSCRIPT_JSON.exists():
        return {}
    with open(TRANSCRIPT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d["scene_id"]: SceneTranscript(**d) for d in data}


def determine_layout_type(region_count: int, clip_screen_type: str) -> str:
    if region_count > LIST_TABLE_REGION_THRESHOLD:
        return "list_table"
    if region_count < MENU_REGION_THRESHOLD:
        return "loading" if clip_screen_type == "loading_transition" else "menu"
    return "form"


def resolve_screen_name(
    scene_id: int,
    ocr_title: str | None,
    ocr_confidence: float,
    clip_confidence: float,
    clip_display_name: str,
    previous_name: str | None,
) -> str:
    if ocr_title and ocr_confidence > OCR_TITLE_CONFIDENCE_FLOOR:
        return ocr_title
    if clip_confidence > CLIP_CONFIDENCE_FLOOR:
        return clip_display_name
    if previous_name:
        return previous_name
    return f"Ekran_{scene_id}"


def enrich_scenes(
    scenes:       list[Scene],
    transcripts:  dict[int, SceneTranscript],
    ocr:          TurkishERPOCR,
    clip:         ERPScreenClassifier,
) -> list[EnrichedScene]:
    results: list[EnrichedScene] = []
    previous_name: str | None = None

    with Progress(
        TextColumn("[ENRICH] Sahne zenginleştirme"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("enrich", total=len(scenes))

        for scene in scenes:
            try:
                ocr_content = ocr.extract_from_frame(scene.keyframe_path)
                region_count = len(ocr_content.regions)
                ocr_confidence = _avg_confidence(ocr_content)

                # list_table ekranlarda label/value eşleştirmesi gürültülü —
                # OCR çalıştı ama eşleştirme sonucu kullanılmıyor.
                ocr_labels = ocr_content.labels if region_count <= LIST_TABLE_REGION_THRESHOLD else {}

                clip_result = clip.classify(scene.keyframe_path, scene_id=scene.scene_id)

                layout_type = determine_layout_type(region_count, clip_result.screen_type)

                transcript = transcripts.get(scene.scene_id)
                transcript_text     = transcript.text if transcript else ""
                transcript_speakers = sorted(transcript.speaker_breakdown.keys()) if transcript else []
                mentioned_actions   = transcript.mentioned_actions if transcript else []
                has_erp_mentions    = transcript.has_erp_mentions if transcript else False

                resolved_name = resolve_screen_name(
                    scene_id           = scene.scene_id,
                    ocr_title           = ocr_content.title,
                    ocr_confidence      = ocr_confidence,
                    clip_confidence     = clip_result.confidence,
                    clip_display_name   = clip_result.display_name,
                    previous_name       = previous_name,
                )
                previous_name = resolved_name

                enriched = EnrichedScene(
                    scene_id          = scene.scene_id,
                    start_time        = scene.start_time,
                    end_time          = scene.end_time,
                    duration          = scene.duration,
                    keyframe_path     = scene.keyframe_path,
                    ocr_title         = ocr_content.title,
                    ocr_buttons       = ocr_content.buttons,
                    ocr_labels        = ocr_labels,
                    ocr_confidence    = ocr_confidence,
                    ocr_region_count  = region_count,
                    layout_type       = layout_type,
                    clip_screen_type  = clip_result.screen_type,
                    clip_display_name = clip_result.display_name,
                    clip_confidence   = clip_result.confidence,
                    clip_top3         = clip_result.top_k,
                    transcript_text     = transcript_text,
                    transcript_speakers = transcript_speakers,
                    mentioned_actions   = mentioned_actions,
                    has_erp_mentions    = has_erp_mentions,
                    resolved_screen_name = resolved_name,
                )
            except Exception as e:
                print(f"[ENRICH HATA] scene_{scene.scene_id:04d}: {e}")
                enriched = EnrichedScene(
                    scene_id=scene.scene_id, start_time=scene.start_time,
                    end_time=scene.end_time, duration=scene.duration,
                    keyframe_path=scene.keyframe_path,
                    ocr_title=None, ocr_buttons=[], ocr_labels={},
                    ocr_confidence=0.0, ocr_region_count=0,
                    layout_type="unknown",
                    clip_screen_type="other_application", clip_display_name="Diğer Uygulama",
                    clip_confidence=0.0,
                    resolved_screen_name=previous_name or f"Ekran_{scene.scene_id}",
                )

            results.append(enriched)
            _print_scene_summary(enriched)
            progress.update(task, advance=1)

    return results


def _print_scene_summary(s: EnrichedScene) -> None:
    name = s.resolved_screen_name[:20].ljust(20)
    if s.layout_type == "list_table":
        extra = f"regions:{s.ocr_region_count}"
    else:
        extra = f"buttons:{s.ocr_buttons[:3]}"
    print(
        f"scene_{s.scene_id:04d} | {s.layout_type:10s} | {name} | "
        f"OCR:{s.ocr_confidence:.2f} CLIP:{s.clip_confidence:.2f} | {extra}"
    )


def _write_enriched_json(scenes: list[EnrichedScene], out_path: Path = ENRICHED_JSON) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in scenes], f, ensure_ascii=False, indent=2)


def run_batch_enrichment(config_path: str = "config.yaml") -> list[EnrichedScene]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    scenes      = _load_scenes()
    transcripts = _load_transcripts()

    ocr_cfg = config.get("ocr", {})
    ocr = TurkishERPOCR(
        use_gpu              = ocr_cfg.get("use_gpu", False),
        confidence_threshold = ocr_cfg.get("confidence_threshold", 0.65),
    )
    clip = ERPScreenClassifier(config)

    enriched = enrich_scenes(scenes, transcripts, ocr, clip)
    _write_enriched_json(enriched)
    print(f"\n[ENRICH] Tamamlandı: {len(enriched)} sahne → {ENRICHED_JSON}")
    return enriched


if __name__ == "__main__":
    run_batch_enrichment()
