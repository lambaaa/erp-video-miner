"""
pipeline/frame_ocr.py
─────────────────────
MODÜL 2 — Her keyframe'den yapısal metin çıkar: başlık, form alanları, butonlar.
PaddleOCR kullanarak ERP ekranlarından yapısal metin çıkarır.

Kurulum:
    pip install paddleocr paddlepaddle opencv-python-headless pillow

Not: İlk çalışmada PaddleOCR modeli indirir (~150MB).
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError(
        "PaddleOCR kurulu değil.\n"
        "  pip install paddleocr paddlepaddle"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  VERİ MODELLERİ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextRegion:
    """OCR'ın tespit ettiği tek bir metin bölgesi."""
    text: str
    confidence: float
    bbox: list              # PaddleOCR formatı: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    region_type: str        # 'title' | 'label' | 'value' | 'button' | 'nav' | 'unknown'
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def __post_init__(self):
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        self.x      = min(xs)
        self.y      = min(ys)
        self.width  = max(xs) - min(xs)
        self.height = max(ys) - min(ys)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class ScreenContent:
    """Bir frame'den çıkarılan tüm yapısal içerik."""
    frame_path: str
    regions:    list[TextRegion]   = field(default_factory=list)
    title:      Optional[str]      = None
    labels:     dict[str, str]     = field(default_factory=dict)  # {"Departman": "Satınalma"}
    buttons:    list[str]          = field(default_factory=list)
    raw_text:   str                = ""
    screen_type: Optional[str]     = None   # CLIP katmanından dolacak


# ══════════════════════════════════════════════════════════════════════════════
#  TÜRKÇE KARAKTER DÜZELTMELERİ
# ══════════════════════════════════════════════════════════════════════════════

# H.264 sıkıştırması + OCR → en sık karıştırılan Türkçe karakterler
_CHAR_FIXES: list[tuple[str, str]] = [
    # OCR sık hatası → doğrusu (regex destekliyor)
    (r'\bI(?=[a-zğüşöçı])', 'İ'),   # "Iade" → "İade"
    (r'(?<=[A-ZĞÜŞÖÇİ])l\b',  'I'),  # sonda l → I (büyük harf bağlamı)
    (r'\b0(?=[A-Za-zğüşıöç])', 'O'),  # 0 → O (kelime başında)
    # Unicode combining karakterler (bazı fontlardan OCR bunu üretir)
    ('\u0067\u0306', 'ğ'),  # g + combining breve
    ('\u0073\u0327', 'ş'),  # s + combining cedilla
    ('\u0075\u0308', 'ü'),  # u + combining diaeresis
    ('\u006f\u0308', 'ö'),
    ('\u0063\u0327', 'ç'),
    ('\u0049\u0307', 'İ'),  # I + combining dot above
]

# ERP sistemlerinde sık görülen Türkçe terimler (fuzzy fix için)
ERP_VOCAB: set[str] = {
    'departman', 'tarih', 'miktar', 'adet', 'birim', 'fiyat', 'toplam',
    'kaydet', 'iptal', 'sil', 'düzenle', 'ekle', 'güncelle', 'onayla',
    'talep', 'sipariş', 'fatura', 'ödeme', 'tedarikçi', 'müşteri',
    'ürün', 'stok', 'depo', 'transfer', 'kategori', 'açıklama',
    'satınalma', 'satış', 'muhasebe', 'finans', 'üretim', 'lojistik',
    'raporlama', 'onay', 'red', 'beklemede', 'tamamlandı', 'iptal edildi',
}

_BUTTON_KEYWORDS: set[str] = {
    'kaydet', 'iptal', 'sil', 'düzenle', 'ekle', 'güncelle', 'onayla',
    'ara', 'temizle', 'kapat', 'geri', 'ileri', 'ok', 'evet', 'hayır',
    'save', 'cancel', 'delete', 'edit', 'add', 'update', 'approve',
    'submit', 'close', 'back', 'next', 'print', 'export', 'yaz', 'gönder',
}


def normalize_turkish(text: str) -> str:
    """
    OCR çıktısındaki Türkçe karakter hatalarını düzelt.

    1. Unicode NFC normalizasyonu (combining karakterleri birleştir)
    2. Bilinen OCR yanlış okumalarını pattern-replace ile düzelt
    3. Baştaki/sondaki boşlukları temizle
    """
    text = unicodedata.normalize('NFC', text)

    for pattern, replacement in _CHAR_FIXES:
        try:
            if pattern.startswith(('(', r'\b', r'(?')):
                text = re.sub(pattern, replacement, text)
            else:
                text = text.replace(pattern, replacement)
        except re.error:
            text = text.replace(pattern, replacement)

    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  ANA OCR SINIFI
# ══════════════════════════════════════════════════════════════════════════════

class TurkishERPOCR:
    """
    ERP eğitim videolarından Türkçe metin çıkaran OCR motoru.

    Tasarım kararları:
    ─ lang='latin': PaddleOCR'ın Latin multilingual modeli, Türkçe UI metinleri
      için 'tr' single-lang modelinden daha stabil. Nedeni: eğitim veri kalitesi.
    ─ Preprocessing önce: video sıkıştırması (H.264) küçük metinleri bulanıklaştırır.
      LANCZOS upscale + CLAHE contrast + unsharp mask → %15-25 accuracy artışı.
    ─ Region classification: position + content heuristikleri → label/value/button ayırımı.
    ─ Label-value pairing: aynı Y eksenindeki en yakın sağ element eşleştirmesi.

    Kullanım:
        ocr     = TurkishERPOCR()
        content = ocr.extract_from_frame("frame_0042.png")
        print(content.title)          # "Satınalma Talebi"
        print(content.labels)         # {"Departman": "Satınalma", "Tarih": "23.06.2026"}
        print(content.buttons)        # ["Kaydet", "İptal"]
    """

    def __init__(
        self,
        use_gpu:              bool  = False,
        confidence_threshold: float = 0.70,
    ):
        self.confidence_threshold = confidence_threshold

        # PaddleOCR 3.x API: lang='tr' (LATIN_LANGS üyesi) → rec_lang='latin' multilingual
        # modeline çözülür; ayrı 'latin' lang kodu artık yok (CLAUDE.md'deki "tr değil,
        # latin" niyeti hâlâ geçerli, sadece API'ye girilen değer değişti).
        # enable_mkldnn=False: bu makinede PaddlePaddle 3.3.1 CPU + oneDNN PIR
        # executor kombinasyonu det modelinde NotImplementedError veriyor.
        self.ocr = PaddleOCR(
            use_textline_orientation = True,    # Döndürülmüş metin tespiti (pop-up'lar için)
            lang                     = 'tr',
            device                   = 'gpu' if use_gpu else 'cpu',
            enable_mkldnn            = False,
            text_det_thresh          = 0.3,     # Düşük → daha fazla alan tespit (gürültü↑)
            text_det_box_thresh      = 0.5,
            text_recognition_batch_size = 8,    # Paralel recognition (hız için)
        )
        print(
            f"[TurkishERPOCR] Hazır │ GPU: {use_gpu} │ "
            f"Eşik: {confidence_threshold}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  PREPROCESSING
    # ──────────────────────────────────────────────────────────────────────────

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Video frame'i OCR için hazırla.

        Neden gerekli:
        - Eğitim videoları genelde 720p + H.264 → küçük metinler bulanık
        - ERP temaları düşük kontrastlı gri tonlar kullanır
        - LANCZOS4 upscale, INTER_CUBIC'ten daha az artefakt üretir

        Pipeline:
            Upscale → Denoise → CLAHE contrast → Unsharp mask
        """
        h, w = frame.shape[:2]

        # 1. Upscale — OCR için minimum 1080p önerilir
        if h < 1080:
            scale  = 1080 / h
            frame  = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # 2. Video sıkıştırma gürültüsünü temizle
        #    (h=5 agresif değil; ERP arayüzü için yeterli)
        frame = cv2.fastNlMeansDenoisingColored(
            frame, None, h=5, hColor=5,
            templateWindowSize=7, searchWindowSize=21,
        )

        # 3. CLAHE — kontrast artır (LAB color space, sadece Luminance kanalı)
        lab               = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch  = cv2.split(lab)
        clahe             = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_ch              = clahe.apply(l_ch)
        frame             = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

        # 4. Unsharp mask — kenarları keskinleştir
        blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
        frame   = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

        return frame

    def preprocess_for_buttons(self, roi: np.ndarray) -> np.ndarray:
        """Buton bölgeleri için binary threshold preprocessing."""
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    def preprocess_for_table(self, roi: np.ndarray) -> np.ndarray:
        """Tablo hücrelerinden grid çizgilerini kaldır."""
        gray       = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((1, 40), np.uint8))
        vertical   = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((40, 1), np.uint8))
        cleaned    = cv2.subtract(gray, cv2.add(horizontal, vertical))
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    # ──────────────────────────────────────────────────────────────────────────
    #  OCR CORE
    # ──────────────────────────────────────────────────────────────────────────

    def _run_ocr(self, image: np.ndarray) -> list[TextRegion]:
        """PaddleOCR çalıştır → TextRegion listesi döndür."""
        result  = self.ocr.predict(image)
        regions = []

        if not result:
            return regions

        page = result[0]
        for poly, text_raw, confidence in zip(
            page['rec_polys'], page['rec_texts'], page['rec_scores']
        ):
            confidence = float(confidence)
            if confidence < self.confidence_threshold:
                continue

            text = normalize_turkish(text_raw)
            if not text:
                continue

            regions.append(TextRegion(
                text        = text,
                confidence  = confidence,
                bbox        = [list(map(float, p)) for p in poly],
                region_type = 'unknown',
            ))

        return regions

    # ──────────────────────────────────────────────────────────────────────────
    #  UI YAPISI ANALİZİ
    # ──────────────────────────────────────────────────────────────────────────

    def classify_regions(
        self,
        regions: list[TextRegion],
        frame_w: int,
        frame_h: int,
    ) -> list[TextRegion]:
        """
        Her TextRegion'ı UI tipine göre sınıflandır.

        ERP ekranı anatomisi (tipik):
        ┌─────────────────────────────────────────────────┐
        │  Başlık/Menü çubuğu  (y < %12)                 │
        ├──────────┬──────────────────────────────────────┤
        │  Nav     │  Form alanları: Label  : Value       │
        │  (x<%18) │  (merkez)                            │
        ├──────────┴──────────────────────────────────────┤
        │  Buton çubuğu  (y > %88)  [Kaydet] [İptal]     │
        └─────────────────────────────────────────────────┘
        """
        for region in regions:
            region.region_type = self._classify_one(region, frame_w, frame_h)
        return regions

    def _classify_one(self, r: TextRegion, fw: int, fh: int) -> str:
        text = r.text.strip()
        cx, cy = r.center

        # Konum
        top_bar     = cy < fh * 0.12
        bottom_bar  = cy > fh * 0.88
        left_panel  = cx < fw * 0.18

        # İçerik
        ends_colon  = text.endswith(':')
        is_short    = len(text) <= 22
        has_digits  = bool(re.search(r'\d', text))
        is_wide     = r.width > fw * 0.28
        t_lower     = text.lower()

        if t_lower in _BUTTON_KEYWORDS or (bottom_bar and is_short and not has_digits):
            return 'button'

        if top_bar:
            return 'title' if r.height > 16 else 'nav'

        if left_panel:
            return 'nav'

        if ends_colon or (is_short and not has_digits and not is_wide):
            return 'label'

        if has_digits or is_wide:
            return 'value'

        return 'unknown'

    def pair_labels_values(self, regions: list[TextRegion]) -> dict[str, str]:
        """
        Label ve value bölgelerini eşleştir.

        Algoritma:
        Her label için → aynı Y ekseninde (±30px tolerans) sağında olan
        en yakın 'value' bölgesini bul. Mesafe eşiği: 500px.

        Neden basit ama yeterli:
        ERP form alanları genelde tek sütunlu düzende veya sabit grid'de.
        Karmaşık iki-sütunlu formlar için bu eşiği genişletebilirsin.
        """
        labels = [r for r in regions if r.region_type == 'label']
        values = [r for r in regions if r.region_type in ('value', 'unknown')]
        pairs  = {}

        for label in labels:
            label_text = label.text.rstrip(':').strip()
            lx, ly     = label.center

            best_value = None
            best_dist  = float('inf')

            for value in values:
                vx, vy = value.center
                # Yatay hizalama kontrolü (±30px Y toleransı)
                if abs(vy - ly) < 30 and vx > lx:
                    dist = vx - lx
                    if dist < best_dist:
                        best_dist  = dist
                        best_value = value

            if best_value and best_dist < 500:
                pairs[label_text] = best_value.text

        return pairs

    # ──────────────────────────────────────────────────────────────────────────
    #  ANA GİRİŞ NOKTASI
    # ──────────────────────────────────────────────────────────────────────────

    def extract_from_frame(
        self,
        frame_input,
        frame_path: str = "",
        pair_labels: bool = True,
    ) -> ScreenContent:
        """
        Tek bir frame'den tam OCR analizi yap.

        Args:
            frame_input : dosya yolu (str/Path) VEYA numpy array (cv2 frame)
            frame_path  : loglama için dosya adı (array verildiğinde)
            pair_labels : label-value eşleştirmesi yapılsın mı. Yoğun tablo
                          ekranlarında (list_table) bu eşleştirme gürültülü
                          ve gereksiz maliyetlidir — çağıran taraf region
                          sayısına göre kapatabilir.

        Returns:
            ScreenContent — title, labels, buttons, regions hepsi dolu
        """
        if isinstance(frame_input, (str, Path)):
            frame_path = str(frame_input)
            frame      = cv2.imread(frame_path)
            if frame is None:
                raise FileNotFoundError(f"Frame yüklenemedi: {frame_path}")
        else:
            frame = frame_input.copy()

        fh, fw = frame.shape[:2]

        # Preprocessing
        preprocessed = self.preprocess(frame)

        # OCR
        regions = self._run_ocr(preprocessed)

        # Sınıflandır
        regions = self.classify_regions(regions, fw, fh)

        # Yapısal analiz
        labels  = self.pair_labels_values(regions) if pair_labels else {}

        # Başlık: en üstte olan 'title' bölgesi
        title = next(
            (r.text for r in sorted(regions, key=lambda r: r.y)
             if r.region_type == 'title'),
            None,
        )

        buttons  = [r.text for r in regions if r.region_type == 'button']
        raw_text = ' | '.join(
            r.text for r in sorted(regions, key=lambda r: (r.y, r.x))
        )

        return ScreenContent(
            frame_path = frame_path,
            regions    = regions,
            title      = title,
            labels     = labels,
            buttons    = buttons,
            raw_text   = raw_text,
        )

    def extract_batch(
        self,
        frames: list,
        show_progress: bool = True,
    ) -> list[ScreenContent]:
        """
        Birden fazla frame'i işle.
        frames: dosya yolu listesi veya numpy array listesi.
        """
        results = []
        total   = len(frames)

        for i, frame in enumerate(frames):
            if show_progress and i % 10 == 0:
                print(f"[OCR] {i}/{total} frame işlendi...")
            try:
                results.append(self.extract_from_frame(frame))
            except Exception as e:
                print(f"[OCR HATA] Frame {i}: {e}")
                results.append(ScreenContent(frame_path=str(frame)))

        if show_progress:
            print(f"[OCR] Tamamlandı: {total} frame.")
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  PİPELİNE ENTEGRASYONU
# ══════════════════════════════════════════════════════════════════════════════

def screen_content_to_xes_events(
    contents:   list[ScreenContent],
    timestamps: list[float],          # saniye cinsinden
    case_id:    str = "session_001",
) -> list[dict]:
    """
    OCR sonuçlarını PM4Py XES event log formatına dönüştür.

    Her buton basışı ve ekran geçişi ayrı bir event olarak kaydedilir.
    Bu formatı pandas DataFrame'e aktarıp pm4py.read_csv() ile okuyabilirsin.

    Dönen dict şeması:
        case:concept:name  → case_id
        concept:name       → aktivite adı
        time:timestamp     → zaman damgası
        screen             → hangi ERP ekranı
        action             → ne yapıldı
        labels_json        → form içeriği (JSON string)
        confidence         → OCR ortalama güven skoru
    """
    events = []

    prev_title = None
    for i, (content, ts) in enumerate(zip(contents, timestamps)):
        screen = content.title or content.screen_type or f"Ekran_{i}"
        conf   = _avg_confidence(content)

        # Ekran geçişi → "Navigate" eventi
        if screen != prev_title:
            events.append({
                'case:concept:name': case_id,
                'concept:name':      f"Navigate → {screen}",
                'time:timestamp':    ts,
                'screen':            screen,
                'action':            'screen_open',
                'labels_json':       _labels_to_str(content.labels),
                'confidence':        round(conf, 3),
                'source':            'scene_change',
            })
            prev_title = screen

        # Her buton → ayrı "Action" eventi
        for button in content.buttons:
            events.append({
                'case:concept:name': case_id,
                'concept:name':      f"{screen} → {button}",
                'time:timestamp':    ts,
                'screen':            screen,
                'action':            button,
                'labels_json':       _labels_to_str(content.labels),
                'confidence':        round(conf, 3),
                'source':            'ocr_button',
            })

    return events


def _avg_confidence(content: ScreenContent) -> float:
    if not content.regions:
        return 0.0
    return sum(r.confidence for r in content.regions) / len(content.regions)


def _labels_to_str(labels: dict) -> str:
    import json
    return json.dumps(labels, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
#  VİDEODAN FRAME ÇIKARMA (bonus: PySceneDetect entegrasyonu olmadan)
# ══════════════════════════════════════════════════════════════════════════════

def extract_keyframes_simple(
    video_path: str,
    output_dir: str = "keyframes",
    diff_threshold: float = 0.08,
    min_gap_seconds: float = 2.0,
) -> list[tuple[str, float]]:
    """
    Basit frame-diff ile sahne değişimlerini tespit et ve kaydet.

    PySceneDetect kurmak istemeyenler için alternatif.
    Diff threshold: 0.08 = %8 piksel değişimi → yeni sahne.

    Returns:
        [(frame_path, timestamp_secs), ...]
    """
    Path(output_dir).mkdir(exist_ok=True)

    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    min_gap  = int(min_gap_seconds * fps)

    prev_gray    = None
    saved        = []
    frame_idx    = 0
    last_saved   = -min_gap

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            diff  = cv2.absdiff(prev_gray, gray)
            score = diff.mean() / 255.0

            if score > diff_threshold and (frame_idx - last_saved) >= min_gap:
                ts       = frame_idx / fps
                out_path = str(Path(output_dir) / f"frame_{frame_idx:06d}.png")
                cv2.imwrite(out_path, frame)
                saved.append((out_path, ts))
                last_saved = frame_idx
                print(f"  [Keyframe] {ts:.1f}s → diff={score:.3f} → {out_path}")

        prev_gray  = gray
        frame_idx += 1

    cap.release()
    print(f"[Keyframe] Toplam {len(saved)} sahne kaydedildi.")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
#  HIZLI TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json

    # ── Video → Keyframe → OCR → XES pipeline testi ──────────────────────────

    if len(sys.argv) > 1 and sys.argv[1].endswith(('.mp4', '.webm', '.mkv', '.avi')):
        print(f"\n[Pipeline] Video: {sys.argv[1]}")
        frames = extract_keyframes_simple(sys.argv[1], output_dir="keyframes_out")

        ocr      = TurkishERPOCR(confidence_threshold=0.65)
        paths    = [f[0] for f in frames]
        times    = [f[1] for f in frames]
        contents = ocr.extract_batch(paths)

        events   = screen_content_to_xes_events(contents, times)
        out_file = "events_xes.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        print(f"\n[Pipeline] {len(events)} event → {out_file}")
        for e in events[:5]:
            print(f"  {e['time:timestamp']:6.1f}s | {e['concept:name']}")

    # ── Tek frame testi ───────────────────────────────────────────────────────
    elif len(sys.argv) > 1:
        ocr     = TurkishERPOCR(confidence_threshold=0.65)
        content = ocr.extract_from_frame(sys.argv[1])

        print(f"\n{'─'*55}")
        print(f"Başlık   : {content.title}")
        print(f"Butonlar : {content.buttons}")
        print(f"\nLabel → Value çiftleri:")
        for k, v in content.labels.items():
            print(f"  {k:28s} → {v}")
        print(f"\nToplam bölge : {len(content.regions)}")
        avg = _avg_confidence(content)
        print(f"Ort. güven   : {avg:.2f}")
        print(f"\nHam metin:\n  {content.raw_text[:300]}")
    else:
        print("Kullanım:")
        print("  python -m pipeline.frame_ocr  video.mp4          # tam pipeline")
        print("  python -m pipeline.frame_ocr  frame_0001.png     # tek frame testi")
