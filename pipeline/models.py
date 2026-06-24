"""
pipeline/models.py
───────────────────
Paylaşılan dataclass'lar. Modüller arası veri sözleşmesini tek yerden
tanımlamak için — ileride diğer modüllerin (Scene, ScreenContent,
ScreenClassification, SceneTranscript) dataclass'ları da buraya taşınacak.
"""

from dataclasses import dataclass, field


@dataclass
class EnrichedScene:
    scene_id:      int
    start_time:    float
    end_time:      float
    duration:      float
    keyframe_path: str

    # OCR
    ocr_title:        str | None
    ocr_buttons:      list[str]
    ocr_labels:       dict[str, str]
    ocr_confidence:   float
    ocr_region_count: int

    # Layout tipi: "form" | "list_table" | "menu" | "loading" | "unknown"
    layout_type: str

    # CLIP
    clip_screen_type:  str
    clip_display_name: str
    clip_confidence:   float
    clip_top3:         list[tuple[str, float]] = field(default_factory=list)

    # Transkript
    transcript_text:     str = ""
    transcript_speakers: list[str] = field(default_factory=list)
    mentioned_actions:   list[str] = field(default_factory=list)
    has_erp_mentions:    bool = False

    # Ekran adı (öncelik: ocr_title > clip_display_name > önceki sahne > fallback)
    resolved_screen_name: str = ""


@dataclass
class ProcessEvent:
    event_id:   str          # "evt_001"
    case_id:    str          # "session_001" (video = 1 oturum)

    # PM4Py XES alanları
    activity:   str          # "Satınalma Talebi → Kaydet"
    timestamp:  float        # saniye
    screen:     str          # resolved_screen_name
    action:     str          # "navigate" | "button_click" | "form_fill" | "screen_view"

    # Bağlam
    form_data:  dict = field(default_factory=dict)   # OCR labels
    transcript: str = ""                              # o andaki transkript

    # Kalite
    confidence: float = 0.0
    signals:    list[str] = field(default_factory=list)
    status:     str = "uncertain"   # "confirmed" | "inferred" | "uncertain"
