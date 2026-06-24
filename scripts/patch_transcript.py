"""
scripts/patch_transcript.py
─────────────────────────────
output/enriched_scenes.json içindeki OCR/CLIP sonuçlarına dokunmadan,
sadece transkript alanlarını output/transcript_aligned.json'daki güncel
(ACTION_WORDS revize edilmiş) verilerle yenile.

Kullanım: python -m scripts.patch_transcript
"""

import json
from pathlib import Path

OUTPUT_DIR      = Path("output")
ENRICHED_JSON   = OUTPUT_DIR / "enriched_scenes.json"
TRANSCRIPT_JSON = OUTPUT_DIR / "transcript_aligned.json"

TRANSCRIPT_FIELDS = (
    "transcript_text", "transcript_speakers", "mentioned_actions", "has_erp_mentions",
)


def patch_transcript_fields() -> int:
    with open(ENRICHED_JSON, "r", encoding="utf-8") as f:
        enriched = json.load(f)

    with open(TRANSCRIPT_JSON, "r", encoding="utf-8") as f:
        transcripts = {t["scene_id"]: t for t in json.load(f)}

    changed = 0
    for scene in enriched:
        transcript = transcripts.get(scene["scene_id"])
        if transcript is None:
            continue

        old_actions = scene["mentioned_actions"]
        new_actions = transcript["mentioned_actions"]
        if old_actions != new_actions:
            changed += 1

        scene["transcript_text"]     = transcript["text"]
        scene["transcript_speakers"] = sorted(transcript["speaker_breakdown"].keys())
        scene["mentioned_actions"]   = new_actions
        scene["has_erp_mentions"]    = transcript["has_erp_mentions"]

    with open(ENRICHED_JSON, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    return changed


if __name__ == "__main__":
    changed = patch_transcript_fields()
    print(f"[PATCH_TRANSCRIPT] {changed} sahnede mentioned_actions değişti.")
    print(f"[PATCH_TRANSCRIPT] {ENRICHED_JSON} güncellendi.")
