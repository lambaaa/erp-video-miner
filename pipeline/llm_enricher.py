"""
pipeline/llm_enricher.py
───────────────────────────
MODÜL 6 — Event listesini özetle ve insan okunabilir süreç dokümantasyonu
üret. SADECE son adımda çalışır, birden fazla çağrı yapma.

Akış: events.json → compress_for_llm() (uncertain'ları at, ekran bazlı
bloklara sıkıştır, ~2000 token sınırı) → Ollama (varsa) → Anthropic (yoksa)
→ ikisi de başarısızsa ham event listesi (CLAUDE.md hata yönetimi kural 4).
"""

import glob
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests
import yaml

OUTPUT_DIR      = Path("output")
EVENTS_JSON     = OUTPUT_DIR / "events.json"
SCENES_JSON     = OUTPUT_DIR / "scenes.json"
PROCESS_SUMMARY = OUTPUT_DIR / "process_summary.md"
HAM_EVENTLER_MD = OUTPUT_DIR / "ham_eventler.md"

MAX_PROMPT_TOKENS  = 2000
TRANSCRIPT_SNIPPET_LEN = 120
ERP_SYSTEM_LABEL   = "F8 Wise (muhasebe modülü)"

BASE_SYSTEM_PROMPT = """\
Sen bir ERP süreç analistisin. F8 Wise muhasebe yazılımı eğitim
videosundan otomatik çıkarılan event log'u analiz et.

KISITLAMALAR:
- Yalnızca event log'daki bilgileri kullan, uydurma
- OCR gürültüsü olan ekran adlarını (×, MM, 0) bağlamdan normalize et
- list_table sahneler = gezinme/inceleme, veri girişi değil

ÇIKTI (Türkçe, markdown):

## Süreç Özeti
[2-3 cümle — ne işlendi, hangi modül, genel akış]

## Süreç Akışı
[Numaralı — sadece confirmed + inferred event'lerden]

## Kullanılan ERP Modülleri
[Ekran adı | Toplam süre | Yapılan işlemler]

## KPI
- Toplam video süresi:
- Tespit edilen ekran geçişi:
- En uzun süre geçirilen ekran:
- Eğitimci tarafından yapılan işlem sayısı:
- Sessiz/geçiş sahnesi sayısı:

## Dikkat Çeken Noktalar
[Uzun bekleme, tekrar eden adım, atlanan ekran]
"""


def _build_system_prompt(video_meta: dict, real_screen_names: list[str]) -> str:
    video_block = (
        "VİDEO BİLGİLERİ (kesin doğru, kullan):\n"
        f"- Toplam süre: {video_meta['duration_label']}\n"
        f"- Sahne sayısı: {video_meta['scene_count']}\n"
        f"- Event sayısı: {video_meta['event_total']} "
        f"({video_meta['confirmed']} confirmed / {video_meta['inferred']} inferred / "
        f"{video_meta['uncertain']} uncertain)\n"
        f"- ERP Sistemi: {video_meta['erp_system']}\n\n"
        "Bu değerleri asla değiştirme veya yeniden hesaplama.\n"
    )
    screen_block = (
        "\n\nKULLANILACAK EKRAN ADLARI (yalnızca bunlar, başka uydurma):\n"
        + "\n".join(f"- {s}" for s in real_screen_names)
        + "\n\nBu listede olmayan hiçbir ekran adı veya uygulama adı yazma."
    )
    return video_block + "\n" + BASE_SYSTEM_PROMPT + screen_block


# ══════════════════════════════════════════════════════════════════════════════
#  COMPRESS — event listesini LLM'e gönderilecek sıkıştırılmış metne çevir
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _fmt_duration_label(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m} dakika {s} saniye"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


NOISE_SCREEN_NAMES = {"MM", "×", "0", "", "Ekran_"}


def normalize_screen_name(name: str) -> str:
    if not name:
        return "Bilinmeyen Ekran"
    if name in NOISE_SCREEN_NAMES:
        return "Bilinmeyen Ekran"
    if name.endswith(":") or len(name) < 3:
        return name.rstrip(":") + " (kısmi OCR)"
    return name


def _group_consecutive_by_screen(events: list[dict]) -> list[list[dict]]:
    blocks: list[list[dict]] = []
    for e in events:
        if blocks and blocks[-1][0]["screen"] == e["screen"]:
            blocks[-1].append(e)
        else:
            blocks.append([e])
    return blocks


def _render_block(idx: int, block: list[dict]) -> str:
    start_ts = block[0]["timestamp"]
    end_ts   = block[-1]["timestamp"]
    screen   = normalize_screen_name(block[0]["screen"])

    actions_display: list[str] = []
    for e in block:
        if e["action"] == "button_click":
            actions_display.append(e["activity"].split(" → ")[-1])
        elif e["action"] == "screen_view":
            actions_display.append("screen_view")
    actions_str = ",".join(dict.fromkeys(actions_display)) if actions_display else "-"

    form_data_merged: dict = {}
    for e in block:
        if e["action"] == "form_fill":
            form_data_merged.update(e["form_data"])

    if form_data_merged:
        extra = ", ".join(f"{k}:{v}" for k, v in list(form_data_merged.items())[:3])
    else:
        transcript_text = next((e["transcript"] for e in block if e["transcript"]), "")
        extra = transcript_text[:TRANSCRIPT_SNIPPET_LEN] if transcript_text else None

    avg_confidence = sum(e["confidence"] for e in block) / len(block)

    line = f"[{idx:03d}] {_fmt_time(start_ts)}-{_fmt_time(end_ts)} | {screen} | {actions_str}"
    if extra:
        line += f" | {extra}"
    line += f" | conf:{avg_confidence:.2f}"
    return line


def _render_compressed_text(events: list[dict]) -> str:
    blocks = _group_consecutive_by_screen(events)
    lines = [_render_block(i + 1, block) for i, block in enumerate(blocks)]
    return "PROCESS EVENTS (ERP session):\n" + "\n".join(lines)


def _filter_and_trim(events: list[dict], max_tokens: int) -> list[dict]:
    """
    1. uncertain event'ler çıkarılır
    2. max_tokens aşılırsa en düşük confidence'lı inferred event'ler atılır
    """
    events = [e for e in events if e["status"] != "uncertain"]

    while True:
        text = _render_compressed_text(events)
        if _estimate_tokens(text) <= max_tokens:
            return events

        inferred = [e for e in events if e["status"] == "inferred"]
        if not inferred:
            return events

        worst = min(inferred, key=lambda e: e["confidence"])
        events.remove(worst)


def _write_ham_eventler_md(events: list[dict], out_path: Path = HAM_EVENTLER_MD) -> None:
    """output/ham_eventler.md — tam (uncertain dahil) ham event listesi. Her zaman
    üretilir (LLM çağrısı başarılı olsa da olmasa da) — compress_for_llm()'in
    side-effect'i."""
    from pipeline.event_builder import TOOLBAR_NOISE

    status_counts = Counter(e["status"] for e in events)
    header = (
        "# Ham Event Listesi\n"
        f"**Oluşturulma:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"**Video:** {_detect_video_label()}\n"
        f"**Toplam event:** {sum(status_counts.values())} "
        f"({status_counts.get('confirmed', 0)} confirmed / "
        f"{status_counts.get('inferred', 0)} inferred / "
        f"{status_counts.get('uncertain', 0)} uncertain)\n"
        f"**Toolbar gürültüsü filtrelendi:** {', '.join(sorted(TOOLBAR_NOISE))}\n"
    )
    lines = [
        f"[{e['event_id']}] {_fmt_time(e['timestamp'])} | {e['screen']} | "
        f"{e['action']} | {e['activity']} | conf:{e['confidence']:.2f} ({e['status']})"
        for e in events
    ]
    content = f"{header}\n---\n\n" + "\n".join(lines) + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def compress_for_llm(events: list[dict], max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """
    Girdi:  events (events.json'daki ham event dict listesi)
    Çıktı:  LLM'e gönderilecek sıkıştırılmış metin (token tasarrufu için)

    Aynı ekranda ardışık event'ler tek bloğa indirilir (bkz. _filter_and_trim).
    Side-effect: output/ham_eventler.md'yi (tam, filtrelenmemiş liste) yazar.
    """
    _write_ham_eventler_md(events)
    return _render_compressed_text(_filter_and_trim(events, max_tokens))


def extract_real_screen_names(events: list[dict], max_tokens: int = MAX_PROMPT_TOKENS) -> list[str]:
    """
    compress_for_llm()'e giden (uncertain'sız, token-bütçesine sığdırılmış) aynı
    event kümesinden normalize edilmiş, tekrarsız ekran adı listesini çıkarır.
    LLM'e "yalnızca bu ekran adlarını kullan" allow-list'i olarak gönderilir.
    """
    trimmed = _filter_and_trim(events, max_tokens)
    return list(dict.fromkeys(normalize_screen_name(e["screen"]) for e in trimmed))


# ══════════════════════════════════════════════════════════════════════════════
#  PROVIDER ÇAĞRILARI
# ══════════════════════════════════════════════════════════════════════════════

def _ollama_available(ollama_url: str) -> bool:
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _call_ollama(user_message: str, model: str, ollama_url: str, temperature: float, system_prompt: str) -> str:
    resp = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "system": system_prompt,
            "prompt": user_message,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=480,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _call_anthropic(user_message: str, system_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text


def _llm_failure_notice() -> str:
    """LLM çalışmadığında process_summary.md'ye yazılacak kısa uyarı (hata yönetimi
    kural 4 — artık ham event listesi değil, bkz. output/ham_eventler.md)."""
    return (
        "> ⚠️ LLM özeti üretilemedi.\n"
        "> Ollama için: `ollama serve` → `ollama pull qwen2.5:7b`\n"
        "> Ham event listesi için: output/ham_eventler.md dosyasına bakın.\n"
    )


def call_llm(compressed_text: str, events: list[dict], config: dict, video_meta: dict) -> tuple[str, str]:
    """
    Girdi:  compressed_text (compress_for_llm çıktısı), events (fallback + ekran adı
            listesi için), config, video_meta (sahne/event sayıları — bkz. run_llm_enrichment)
    Çıktı:  (markdown_metni, kullanılan_model_etiketi)
    """
    llm_cfg = config.get("llm", {})
    if not llm_cfg.get("enabled", True):
        return _llm_failure_notice(), "devre dışı (llm.enabled=false)"

    ollama_url   = llm_cfg.get("ollama_url", "http://localhost:11434")
    ollama_model = llm_cfg.get("model", "qwen2.5:7b")
    temperature  = llm_cfg.get("temperature", 0.1)

    real_screen_names = extract_real_screen_names(events)
    system_prompt = _build_system_prompt(video_meta, real_screen_names)

    if _ollama_available(ollama_url):
        try:
            return (
                _call_ollama(compressed_text, ollama_model, ollama_url, temperature, system_prompt),
                f"ollama/{ollama_model}",
            )
        except Exception as e:
            print(f"[LLM HATA] Ollama çağrısı başarısız: {e}")

    try:
        return _call_anthropic(compressed_text, system_prompt), "anthropic/claude-haiku-4-5-20251001"
    except Exception as e:
        print(f"[LLM HATA] Anthropic çağrısı başarısız: {e}")

    print("[LLM HATA] Hiçbir provider çalışmadı.")
    return _llm_failure_notice(), "LLM bağlantısı kurulamadı"


def enrich_events(events: list[dict], config: dict) -> str:
    """
    Girdi:  events (Modül 5 çıktısı, dict listesi), config (config.yaml'dan llm bölümü)
    Çıktı:  process_summary.md içeriği (markdown string, metadata başlığı hariç)

    Not: output/scenes.json'a erişimi olmadığı için video_meta burada events'ten
    yaklaşık olarak türetilir (run_llm_enrichment'taki gibi tam doğru değildir).
    """
    status_counts = Counter(e["status"] for e in events)
    video_meta = {
        "duration_label": _fmt_duration_label(max((e["timestamp"] for e in events), default=0)),
        "scene_count": "bilinmiyor",
        "event_total": sum(status_counts.values()),
        "confirmed": status_counts.get("confirmed", 0),
        "inferred": status_counts.get("inferred", 0),
        "uncertain": status_counts.get("uncertain", 0),
        "erp_system": ERP_SYSTEM_LABEL,
    }

    compressed_text = compress_for_llm(events)
    markdown, _ = call_llm(
        compressed_text, events,
        {"llm": config} if "enabled" in config else config,
        video_meta,
    )
    return markdown


# ══════════════════════════════════════════════════════════════════════════════
#  ÇIKTI DOSYASI
# ══════════════════════════════════════════════════════════════════════════════

def _detect_video_label() -> str:
    matches = glob.glob("docs/*.mp4")
    if not matches:
        return "F8 Wise Muhasebe Eğitimi"
    stem = Path(matches[0]).stem
    m = re.match(r"(.+?)-(\d{8})", stem)
    return f"{m.group(1)}-{m.group(2)}" if m else stem


def _write_process_summary(
    markdown: str,
    model_label: str,
    out_path: Path = PROCESS_SUMMARY,
) -> str:
    header = (
        "# Süreç Özeti\n"
        f"**Oluşturulma:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"**Video:** {_detect_video_label()}\n"
        f"**Model:** {model_label}\n"
        f"**Ham event listesi:** output/ham_eventler.md\n"
    )
    content = f"{header}\n{markdown}\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def run_llm_enrichment(config_path: str = "config.yaml") -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(EVENTS_JSON, "r", encoding="utf-8") as f:
        events = json.load(f)

    status_counts = Counter(e["status"] for e in events)

    try:
        with open(SCENES_JSON, "r", encoding="utf-8") as f:
            scenes = json.load(f)
        duration_label = _fmt_duration_label(scenes[-1]["end_time"])
        scene_count = len(scenes)
    except (FileNotFoundError, IndexError):
        duration_label = _fmt_duration_label(max((e["timestamp"] for e in events), default=0))
        scene_count = "bilinmiyor"

    video_meta = {
        "duration_label": duration_label,
        "scene_count": scene_count,
        "event_total": sum(status_counts.values()),
        "confirmed": status_counts.get("confirmed", 0),
        "inferred": status_counts.get("inferred", 0),
        "uncertain": status_counts.get("uncertain", 0),
        "erp_system": ERP_SYSTEM_LABEL,
    }

    compressed_text = compress_for_llm(events)
    markdown, model_label = call_llm(compressed_text, events, config, video_meta)

    content = _write_process_summary(markdown, model_label)

    print(f"[LLM_ENRICHER] Model: {model_label}")
    print(f"[LLM_ENRICHER] → {PROCESS_SUMMARY}\n")
    print(content)
    return content


if __name__ == "__main__":
    run_llm_enrichment()
