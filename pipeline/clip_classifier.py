"""
pipeline/clip_classifier.py
─────────────────────────────
MODÜL 3 — Her frame'i CLIP ile ERP ekran tiplerine göre zero-shot sınıflandır.
Training gerekmez — metin prompt'ları ile çalışır.

Not: F8 Wise muhasebe eğitimi videosunda gözlenen ekranlara göre prompt seti
(invoice_entry, invoice_list, ...) genel "purchase_request" setinden farklılaştırıldı.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

ERP_SCREEN_PROMPTS: dict[str, str] = {
    "invoice_entry":      "an invoice entry form in an ERP accounting system",
    "invoice_list":       "a list or table of invoices in an ERP system",
    "approval_workflow":  "an approval workflow or authorization screen",
    "accounting_entry":   "an accounting journal or general ledger entry screen",
    "vendor_list":        "a vendor or supplier list table in ERP",
    "payment_screen":     "a payment matching or payment entry screen",
    "report_screen":      "a report or analytics dashboard screen",
    "menu_navigation":    "a main menu or navigation screen",
    "settings_screen":    "a settings or configuration screen",
    "other_application":  "a non-ERP application window or desktop",
    "loading_transition":  "a loading or transition screen",
}

DISPLAY_NAMES: dict[str, str] = {
    "invoice_entry":      "Fatura Girişi",
    "invoice_list":       "Fatura Listesi",
    "approval_workflow":  "Onay Süreci",
    "accounting_entry":   "Muhasebe Fişi",
    "vendor_list":        "Tedarikçi Listesi",
    "payment_screen":     "Ödeme Eşleştirme",
    "report_screen":      "Raporlama",
    "menu_navigation":    "Ana Menü",
    "settings_screen":    "Ayarlar",
    "other_application":  "Diğer Uygulama",
    "loading_transition": "Yükleniyor",
}


@dataclass
class ScreenClassification:
    scene_id:     int
    screen_type:  str     # "invoice_entry" | "vendor_list" | ...
    display_name: str     # Türkçe görünen ad
    confidence:   float
    top_k:        list[tuple[str, float]] = field(default_factory=list)


class ERPScreenClassifier:
    """
    CLIP zero-shot sınıflandırıcı. Model bir kez yüklenir, tüm frame'ler için
    tekrar kullanılır (her çağrıda yeniden yükleme maliyetinden kaçınmak için).
    """

    def __init__(self, config: dict):
        clip_cfg        = config.get("clip", config)
        model_name       = clip_cfg.get("model", "openai/clip-vit-base-patch32")
        cache_dir        = clip_cfg.get("cache_dir", "./models")
        self.top_k       = clip_cfg.get("top_k", 3)
        self.batch_size  = clip_cfg.get("batch_size", 8)

        self.prompt_keys   = list(ERP_SCREEN_PROMPTS.keys())
        self.prompt_texts  = list(ERP_SCREEN_PROMPTS.values())

        self.model     = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()

        print(f"[ERPScreenClassifier] Hazır │ model: {model_name}")

    def classify(self, frame_path: str, scene_id: int = 0) -> ScreenClassification:
        from PIL import Image

        image  = Image.open(frame_path).convert("RGB")
        inputs = self.processor(
            text=self.prompt_texts, images=image,
            return_tensors="pt", padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs   = outputs.logits_per_image.softmax(dim=1)[0].tolist()

        ranked = sorted(zip(self.prompt_keys, probs), key=lambda kv: kv[1], reverse=True)

        screen_type, confidence = ranked[0]
        top_k = ranked[: self.top_k]

        return ScreenClassification(
            scene_id     = scene_id,
            screen_type  = screen_type,
            display_name = DISPLAY_NAMES.get(screen_type, screen_type),
            confidence   = float(confidence),
            top_k        = [(k, float(v)) for k, v in top_k],
        )

    def classify_batch(
        self, frame_paths: list[str], show_progress: bool = True,
    ) -> list[ScreenClassification]:
        results = []
        total   = len(frame_paths)

        for i in range(0, total, self.batch_size):
            batch = frame_paths[i : i + self.batch_size]
            if show_progress:
                print(f"[CLIP] {i}/{total} frame işlendi...")
            for j, path in enumerate(batch):
                try:
                    results.append(self.classify(path, scene_id=i + j))
                except Exception as e:
                    print(f"[CLIP HATA] {path}: {e}")
                    results.append(ScreenClassification(
                        scene_id=i + j, screen_type="other_application",
                        display_name=DISPLAY_NAMES["other_application"],
                        confidence=0.0,
                    ))

        if show_progress:
            print(f"[CLIP] Tamamlandı: {total} frame.")
        return results


# Geriye dönük uyumluluk: modül seviyesinde tek-frame fonksiyon arayüzü.
_classifier_cache: ERPScreenClassifier | None = None


def classify_screen(frame_path: str, config: dict) -> ScreenClassification:
    """
    Girdi:  frame_path, config (config.yaml'dan clip bölümü)
    Çıktı:  ScreenClassification
    """
    global _classifier_cache
    if _classifier_cache is None:
        _classifier_cache = ERPScreenClassifier(config)
    return _classifier_cache.classify(frame_path)


if __name__ == "__main__":
    import sys

    config = {"clip": {
        "model": "openai/clip-vit-base-patch32",
        "cache_dir": "./models",
        "top_k": 3,
    }}

    clf = ERPScreenClassifier(config)

    paths = sys.argv[1:] or ["output/keyframes/scene_0000.png"]
    for path in paths:
        result = clf.classify(path)
        print(f"\n{'─'*55}")
        print(f"Frame      : {Path(path).name}")
        print(f"Ekran tipi : {result.screen_type} ({result.display_name})")
        print(f"Confidence : {result.confidence:.3f}")
        print("Top-k      :")
        for k, v in result.top_k:
            print(f"  {DISPLAY_NAMES.get(k, k):20s} {v:.3f}")
