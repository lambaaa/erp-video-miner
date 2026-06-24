"""
pipeline/event_builder.py
────────────────────────────
MODÜL 5 — output/enriched_scenes.json'ı (OCR + CLIP + transkript birleşmiş
sahneler) State→Action kurallarıyla event log'a dönüştür.

Ağırlıklar CLAUDE.md "REVİZE KARARLAR" bölümündeki revize değerlerini kullanır
(CLIP zayıf sinyal → düşük ağırlık, OCR title → yeni güçlü sinyal).
layout_type == "list_table" sahnelerde tek "screen_view" event'i üretilir,
buton bazlı "button_click" üretilmez (tablo hücreleri buton değildir).
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from pipeline.models import EnrichedScene, ProcessEvent

OUTPUT_DIR      = Path("output")
ENRICHED_JSON   = OUTPUT_DIR / "enriched_scenes.json"
EVENTS_JSON     = OUTPUT_DIR / "events.json"
AUDIT_TRAIL_CSV = OUTPUT_DIR / "audit_trail.csv"

DEFAULT_WEIGHTS = {
    "screen_change":      0.20,
    "ocr_title_detected": 0.35,
    "ocr_button":         0.30,
    "transcript_action":  0.25,
    "clip_signal":        0.10,
    "form_data_change":   0.20,
}
DEFAULT_THRESHOLDS = {"confirmed": 0.70, "inferred": 0.40}
DEFAULT_OCR_TITLE_FLOOR  = 0.65
DEFAULT_CLIP_FLOOR       = 0.55
DEFAULT_SKIP_LAYOUTS     = {"loading"}
MAX_BUTTONS_PER_SCENE    = 5

TOOLBAR_NOISE = {
    # Standart F8 Wise toolbar butonları
    "Görevler", "Ekler", "Çıktı", "Çikti",
    "Yardım", "Yardim", "Sistem Bilgisi", "Geri", "İleri",
    # Truncated OCR okumaları (bunlar toolbar'ın kesik halleri)
    "evler", "örevler", "ptal Et", "ler", "Wi",
    # Tek karakter / sembol
    "人", "Y",
}


def _load_enriched_scenes() -> list[EnrichedScene]:
    with open(ENRICHED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EnrichedScene(**d) for d in data]


def _status_from_confidence(confidence: float, thresholds: dict) -> str:
    if confidence >= thresholds["confirmed"]:
        return "confirmed"
    if confidence >= thresholds["inferred"]:
        return "inferred"
    return "uncertain"


def _identity_signals(
    scene: EnrichedScene, weights: dict, ocr_title_floor: float, clip_floor: float,
) -> tuple[float, list[str]]:
    """OCR title + CLIP + transkript aksiyon sinyallerinden ortak confidence katkısı."""
    confidence = 0.0
    signals: list[str] = []

    if scene.ocr_title and scene.ocr_confidence > ocr_title_floor:
        confidence += weights["ocr_title_detected"]
        signals.append("ocr_title")

    if scene.clip_confidence > clip_floor:
        confidence += weights["clip_signal"]
        signals.append("clip_signal")

    if scene.mentioned_actions:
        confidence += weights["transcript_action"]
        signals.append("transcript_action")

    return confidence, signals


def build_events(
    scenes: list[EnrichedScene],
    config: dict,
    case_id: str = "session_001",
) -> list[ProcessEvent]:
    """
    Girdi:  scenes (output/enriched_scenes.json'dan EnrichedScene listesi), config
    Çıktı:  list[ProcessEvent]
    """
    eb_cfg          = config.get("event_builder", {})
    weights         = {**DEFAULT_WEIGHTS, **eb_cfg.get("signal_weights", {})}
    thresholds      = {**DEFAULT_THRESHOLDS, **eb_cfg.get("status_thresholds", {})}
    ocr_title_floor = eb_cfg.get("ocr_title_confidence_floor", DEFAULT_OCR_TITLE_FLOOR)
    clip_floor      = eb_cfg.get("clip_confidence_floor", DEFAULT_CLIP_FLOOR)
    skip_layouts    = set(eb_cfg.get("skip_layout_types", DEFAULT_SKIP_LAYOUTS))

    events: list[ProcessEvent] = []
    event_seq = 0

    previous_screen_name: str | None = None
    previous_labels: dict = {}
    previous_buttons: set[str] = set()

    def next_event(**kwargs) -> ProcessEvent:
        nonlocal event_seq
        event_seq += 1
        confidence = kwargs.pop("confidence")
        return ProcessEvent(
            event_id   = f"evt_{event_seq:03d}",
            case_id    = case_id,
            confidence = round(confidence, 3),
            status     = _status_from_confidence(confidence, thresholds),
            **kwargs,
        )

    for scene in scenes:
        if scene.layout_type in skip_layouts:
            continue

        identity_conf, identity_signals = _identity_signals(scene, weights, ocr_title_floor, clip_floor)

        # Kural 1 — Ekran geçişi
        if previous_screen_name is not None and scene.resolved_screen_name != previous_screen_name:
            confidence = weights["screen_change"] + identity_conf
            events.append(next_event(
                activity   = f"{previous_screen_name} → {scene.resolved_screen_name}",
                timestamp  = scene.start_time,
                screen     = scene.resolved_screen_name,
                action     = "navigate",
                form_data  = {},
                transcript = scene.transcript_text,
                confidence = confidence,
                signals    = ["screen_change"] + identity_signals,
            ))

        if scene.layout_type == "list_table":
            # Kural 2 (revize) — tablo/liste ekranı: tek screen_view event'i,
            # buton bazlı event yok (tablo hücreleri buton değildir).
            confidence = identity_conf
            events.append(next_event(
                activity   = f"{scene.resolved_screen_name} (liste görünümü, {scene.ocr_region_count} OCR bölgesi)",
                timestamp  = scene.start_time,
                screen     = scene.resolved_screen_name,
                action     = "screen_view",
                form_data  = {},
                transcript = scene.transcript_text,
                confidence = confidence,
                signals    = identity_signals,
            ))
            previous_buttons = set()
        else:
            # Kural 2 — OCR buton tespiti. Aynı toolbar butonları (Görevler/Ekler/...)
            # her sahnede tekrar görünüyor; sadece önceki sahnede olmayan
            # (yeni beliren) butonlar gerçek bir aksiyon adayı sayılır.
            current_buttons = scene.ocr_buttons[:MAX_BUTTONS_PER_SCENE]
            new_buttons = [b for b in current_buttons if b not in previous_buttons]

            for button in new_buttons:
                if button in TOOLBAR_NOISE:
                    continue

                confidence = weights["ocr_button"] + identity_conf
                events.append(next_event(
                    activity   = f"{scene.resolved_screen_name} → {button}",
                    timestamp  = scene.start_time,
                    screen     = scene.resolved_screen_name,
                    action     = "button_click",
                    form_data  = {},
                    transcript = scene.transcript_text,
                    confidence = confidence,
                    signals    = ["ocr_button"] + identity_signals,
                ))

            # Kural 4 — Form değişimi (aynı ekranda kalındıysa)
            if scene.layout_type == "form" and scene.resolved_screen_name == previous_screen_name:
                changed_fields = {
                    k: v for k, v in scene.ocr_labels.items()
                    if previous_labels.get(k) != v
                }
                if changed_fields:
                    confidence = weights["form_data_change"] + identity_conf
                    events.append(next_event(
                        activity   = f"{scene.resolved_screen_name} → {', '.join(changed_fields)} güncellendi",
                        timestamp  = scene.start_time,
                        screen     = scene.resolved_screen_name,
                        action     = "form_fill",
                        form_data  = changed_fields,
                        transcript = scene.transcript_text,
                        confidence = confidence,
                        signals    = ["form_data_change"] + identity_signals,
                    ))

            previous_buttons = set(current_buttons)

        previous_screen_name = scene.resolved_screen_name
        previous_labels      = scene.ocr_labels

    return events


def _write_events_json(events: list[ProcessEvent], out_path: Path = EVENTS_JSON) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in events], f, ensure_ascii=False, indent=2)


def _write_audit_trail_csv(events: list[ProcessEvent], out_path: Path = AUDIT_TRAIL_CSV) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id", "timestamp", "activity", "screen", "action", "confidence", "status",
            "form_data_json", "signals_json", "transcript_excerpt",
        ])
        for e in events:
            writer.writerow([
                e.event_id, e.timestamp, e.activity, e.screen, e.action, e.confidence, e.status,
                json.dumps(e.form_data, ensure_ascii=False),
                json.dumps(e.signals, ensure_ascii=False),
                e.transcript[:120],
            ])


def run_event_builder(config_path: str = "config.yaml") -> list[ProcessEvent]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    scenes = _load_enriched_scenes()
    events = build_events(scenes, config)

    _write_events_json(events)
    _write_audit_trail_csv(events)

    from collections import Counter
    status_counts = Counter(e.status for e in events)
    print(f"[EVENT_BUILDER] {len(scenes)} sahneden {len(events)} event üretildi.")
    print(f"[EVENT_BUILDER] Status dağılımı: {dict(status_counts)}")
    print(f"[EVENT_BUILDER] → {EVENTS_JSON}, {AUDIT_TRAIL_CSV}")
    return events


if __name__ == "__main__":
    run_event_builder()
