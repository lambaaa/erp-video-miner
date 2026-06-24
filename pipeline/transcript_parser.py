"""
pipeline/transcript_parser.py
────────────────────────────────
MODÜL 4 — VTT/SRT veya Teams DOCX transkriptini parse et ve sahnelerle
zaman eşleştir. Transkript yoksa Whisper small ile ses dosyasından üret.

Teams DOCX export formatı (her utterance bir paragraf):
    "\nNur ALLAHVERDİ   0:34\nMetin satırı 1.\nMetin satırı 2."
"""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pipeline.scene_detector import Scene

try:
    import docx
except ImportError:
    docx = None

OUTPUT_DIR          = Path("output")
TRANSCRIPT_ALIGNED  = OUTPUT_DIR / "transcript_aligned.json"

ACTION_WORDS = {
    # Orijinal (teknik)
    "kaydet", "kaydediyoruz", "kaydettik",
    "tıklıyoruz", "tıklayalım", "tıkladım",
    "açıyoruz", "açalım", "açıldı",
    "seçiyoruz", "seçelim", "seçtim",
    "giriyoruz", "girelim", "girdim",
    "onaylıyoruz", "onayladık",
    # Konuşma dili (eğitimci)
    "bakıyoruz", "bakalım", "bakın",
    "gidiyoruz", "gidelim", "gittik",
    "geçiyoruz", "geçelim", "geçtik",
    "dolduruyoruz", "dolduralım", "doldurabilirsiniz",
    "yapıyoruz", "yapalım", "yaptık",
    "oluşturuyoruz", "oluşturalım", "oluşturmamız",
    "tanımlıyoruz", "tanımlayalım",
    "göndereceğiz", "gönderelim", "gönderiyoruz",
    "gösteriyorum", "görelim", "görüyoruz",
    "klikleyelim", "klik", "sağ klik",
    "seçeneği", "butonuna", "ekranına",
    "başlıyoruz", "başlayalım",
}


ERP_TERMS = {
    "erp", "fatura", "sipariş", "stok", "tedarikçi", "muhasebe",
    "fiş", "cari", "mutabakat", "onay rotası", "vergi kodu",
    "satınalma", "f8", "wise",
}

_SPEAKER_LINE_RE = re.compile(r"^(?P<speaker>.+?)\s{2,}(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)$")


@dataclass
class SceneTranscript:
    scene_id:           int
    text:               str                       # o sahneye ait transkript metni
    segments:           list[dict] = field(default_factory=list)
    has_erp_mentions:   bool = False
    mentioned_actions:  list[str] = field(default_factory=list)
    speaker_breakdown:  dict[str, str] = field(default_factory=dict)


def _timestamp_to_seconds(ts: str) -> float:
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 2:
        m, s = parts
        return float(m * 60 + s)
    h, m, s = parts
    return float(h * 3600 + m * 60 + s)


def parse_docx_transcript(path: str | Path) -> list[dict]:
    """
    Girdi:  Teams DOCX transkript dosya yolu
    Çıktı:  list[dict] — {"speaker", "start_sec", "text", "is_trainer"}
    """
    if docx is None:
        raise ImportError("python-docx kurulu değil.\n  pip install python-docx")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transkript bulunamadı: {path}")

    document = docx.Document(str(path))

    raw: list[tuple[str, float, str]] = []
    for paragraph in document.paragraphs:
        lines = [ln for ln in paragraph.text.strip("\n").split("\n") if ln.strip()]
        if not lines:
            continue
        m = _SPEAKER_LINE_RE.match(lines[0].strip())
        if not m:
            continue
        speaker = m.group("speaker").strip()
        start_sec = _timestamp_to_seconds(m.group("ts"))
        text = " ".join(ln.strip() for ln in lines[1:]).strip()
        if not text:
            continue
        raw.append((speaker, start_sec, text))

    speaker_counts = Counter(speaker for speaker, _, _ in raw)
    max_count = max(speaker_counts.values(), default=0)
    trainers = {s for s, c in speaker_counts.items() if c == max_count}
    is_tie = len(trainers) > 1

    utterances = []
    for speaker, start_sec, text in raw:
        utterances.append({
            "speaker": speaker,
            "start_sec": start_sec,
            "text": text,
            "is_trainer": (not is_tie) and (speaker in trainers),
        })
    return utterances


def align_to_scenes(
    utterances: list[dict],
    scenes:     list[Scene],
    config:     dict,
) -> list[SceneTranscript]:
    """
    Girdi:  utterances (parse_docx_transcript çıktısı), scenes, config
    Çıktı:  list[SceneTranscript] — her sahne için bir kayıt
    """
    alignment_window = config.get("alignment_window", 3.0)
    results: list[SceneTranscript] = []

    for scene in scenes:
        window_start = scene.start_time - alignment_window
        window_end = scene.end_time + alignment_window
        matched = [u for u in utterances if window_start <= u["start_sec"] <= window_end]
        matched.sort(key=lambda u: u["start_sec"])

        text = " ".join(u["text"] for u in matched)
        segments = [
            {"start": u["start_sec"], "text": u["text"], "speaker": u["speaker"]}
            for u in matched
        ]

        speaker_breakdown: dict[str, str] = {}
        for u in matched:
            if u["speaker"] in speaker_breakdown:
                speaker_breakdown[u["speaker"]] += " " + u["text"]
            else:
                speaker_breakdown[u["speaker"]] = u["text"]

        lowered = text.lower()
        mentioned_actions = sorted(w for w in ACTION_WORDS if w in lowered)
        has_erp_mentions = any(term in lowered for term in ERP_TERMS)

        results.append(SceneTranscript(
            scene_id=scene.scene_id,
            text=text,
            segments=segments,
            has_erp_mentions=has_erp_mentions,
            mentioned_actions=mentioned_actions,
            speaker_breakdown=speaker_breakdown,
        ))

    return results


def _write_transcript_aligned(scene_transcripts: list[SceneTranscript], out_path: Path = TRANSCRIPT_ALIGNED) -> None:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in scene_transcripts], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[TRANSCRIPT HATA] transcript_aligned.json yazılamadı: {e}")


def parse_transcript(
    transcript_path: str | Path | None,
    video_path:      str | Path,
    scenes:          list[Scene],
    config:          dict,
) -> list[SceneTranscript]:
    """
    Girdi:  transcript_path (yoksa None → Whisper kullanılır), video_path,
            scenes (Modül 1 çıktısı), config
    Çıktı:  list[SceneTranscript]
    """
    empty_results = [SceneTranscript(scene_id=s.scene_id, text="") for s in scenes]

    if transcript_path is None:
        print("[TRANSCRIPT HATA] Whisper transkripsiyon henüz desteklenmiyor (TODO).")
        _write_transcript_aligned(empty_results)
        return empty_results

    transcript_path = Path(transcript_path)
    suffix = transcript_path.suffix.lower()

    try:
        if suffix == ".docx":
            utterances = parse_docx_transcript(transcript_path)
            results = align_to_scenes(utterances, scenes, config)
        else:
            print(f"[TRANSCRIPT HATA] '{suffix}' formatı henüz desteklenmiyor (VTT/SRT TODO).")
            results = empty_results
    except Exception as e:
        print(f"[TRANSCRIPT HATA] Transkript işlenemedi: {e}")
        results = empty_results

    _write_transcript_aligned(results)
    return results


if __name__ == "__main__":
    import sys

    import yaml

    if len(sys.argv) < 2:
        print("Kullanım: python -m pipeline.transcript_parser transkript.docx")
        sys.exit(1)

    docx_path = sys.argv[1]

    scenes_json_path = Path("output") / "scenes.json"
    if not scenes_json_path.exists():
        print(f"[TRANSCRIPT HATA] {scenes_json_path} bulunamadı. Önce scene_detector çalıştırın.")
        sys.exit(1)

    with open(scenes_json_path, "r", encoding="utf-8") as f:
        scene_dicts = json.load(f)
    scenes = [Scene(**d) for d in scene_dicts]

    cfg = {"alignment_window": 3.0, "min_segment_duration": 0.5}
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("transcript", cfg)
    except FileNotFoundError:
        pass

    utterances = parse_docx_transcript(docx_path)
    scene_transcripts = align_to_scenes(utterances, scenes, cfg)
    _write_transcript_aligned(scene_transcripts)

    speaker_counts = Counter(u["speaker"] for u in utterances)
    max_count = max(speaker_counts.values(), default=0)
    trainers = [s for s, c in speaker_counts.items() if c == max_count]
    trainer_label = trainers[0] if len(trainers) == 1 else "belirlenemedi (esit konusma)"

    n_with_text = sum(1 for t in scene_transcripts if t.text.strip())

    print(f"\n[TRANSCRIPT] Toplam {len(utterances)} utterance bulundu.")
    print(f"[TRANSCRIPT] Tespit edilen egitimci: {trainer_label}")
    print("[TRANSCRIPT] Ilk 3 utterance:")
    for u in utterances[:3]:
        print(f"  {u['speaker']!r} | {u['start_sec']:.1f}s | {u['text'][:80]!r}")
    print(f"[TRANSCRIPT] {n_with_text}/{len(scene_transcripts)} sahnede transkript var.")
