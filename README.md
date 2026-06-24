# ERP Video Process Miner

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-CLIP-yellow?style=for-the-badge)](https://huggingface.co/docs/transformers/model_doc/clip)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-0062B0?style=for-the-badge&logo=paddlepaddle&logoColor=white)](https://github.com/PaddlePaddle/PaddleOCR)
[![Whisper](https://img.shields.io/badge/OpenAI%20Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Ollama](https://img.shields.io/badge/Ollama-qwen2.5-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **Automatic audit trail and process documentation from ERP training videos — without access to ERP logs.**

[🇹🇷 Türkçe sürüm](README.tr.md)

---

## The Problem

ERP implementations produce hours of training recordings. Inside those videos is a complete record of how the system is actually used — which screens, which workflows, which sequences. But extracting that knowledge manually is slow, inconsistent, and almost never done.

At the same time, getting structured event data *out of* most ERP systems requires DBA access, custom log configurations, or expensive process mining licenses. Small teams don't have any of those.

This project takes a different path: **treat the training video itself as the data source.**

---

## How It Works

```
Video (MP4)
  │
  ▼  [1] Scene Detection      deterministic · PySceneDetect
  │     detects screen changes, extracts keyframes
  │
  ▼  [2] OCR                  deterministic · PaddleOCR
  │     reads screen titles, form fields, button labels
  │
  ▼  [3] CLIP Classification  local model · no API call
  │     zero-shot screen type identification
  │
  ▼  [4] Transcript Alignment deterministic · python-docx / Whisper
  │     maps speech to screen segments, detects action keywords
  │
  ▼  [5] Event Builder        rule-based · no model
  │     state → action inference, confidence scoring
  │     output: 136 events (confirmed / inferred / uncertain)
  │
  ▼  [6] LLM Enricher         LAST STEP ONLY · Ollama / Anthropic
        input: ~120 filtered events (~1300 tokens)
        output: structured process summary in Markdown
```

**Core design rule:** The LLM never sees raw frames, raw OCR text, or video. It receives only a compressed, filtered event log — typically under 2000 tokens per session.

---

## Output

| File | Description |
|------|-------------|
| `output/events.json` | Structured process events with confidence scores |
| `output/audit_trail.csv` | Flat audit trail (PM4Py / XES compatible) |
| `output/ham_eventler.md` | Full raw event list — always generated, even without LLM |
| `output/process_summary.md` | LLM-generated process summary |
| `output/enriched_scenes.json` | Cached OCR + CLIP + transcript data per scene |

---

## Quick Start

```bash
git clone https://github.com/yourname/erp-video-miner
cd erp-video-miner
pip install -r requirements.txt

# With transcript (Teams DOCX export)
python main.py --video input.mp4 --transcript transcript.docx

# Without transcript (Whisper fallback)
python main.py --video input.mp4

# Skip OCR re-processing (enriched_scenes.json already exists)
python main.py --video input.mp4 --transcript transcript.docx --skip-ocr

# No LLM — deterministic pipeline only
python main.py --video input.mp4 --no-llm
```

For local LLM (recommended, free):
```bash
# Install Ollama: https://ollama.com/download
ollama pull qwen2.5:7b
ollama serve
```

All thresholds and model choices are in `config.yaml` — no magic numbers in code.

---

## POC Results

Tested on a **76-minute F8 Wise (IFS) accounting module training video** with Teams DOCX transcript.

| Metric | Result |
|--------|--------|
| Scenes detected | 52 |
| Events generated | 136 |
| Confirmed events | 49 (36%) |
| Inferred events | 72 (53%) |
| Uncertain events | 15 (11%) |
| Scenes with transcript signal | 27 / 52 |
| Scenes without transcript (silent navigation) | 25 / 52 |
| LLM input (after filtering) | 121 events / ~1300 tokens |
| Total processing time (CPU, no GPU) | ~2.5 hours* |

*The 2.5-hour runtime is a one-time cost. `enriched_scenes.json` is cached — subsequent runs (event rebuild, LLM re-run, config tuning) complete in seconds.

**Correctly identified ERP screens:** Gelen Fatura (Incoming Invoice), Muhasebe Kuralları (Accounting Rules), FEKA İşleme Kontrol Detayları, Finans, Tedarikçi (Vendor) screens.

---

## Technology

| Layer | Tool | Why |
|-------|------|-----|
| Scene detection | PySceneDetect | ContentDetector tuned for UI screen changes |
| OCR | PaddleOCR (`lang='latin'`) | Better than `lang='tr'` for mixed Turkish/English ERP UI text |
| Screen classification | CLIP `vit-base-patch32` | Zero-shot — no training data needed |
| Transcription (fallback) | Whisper `small` | Better Turkish accuracy than `base`; word-level timestamps |
| Transcript parsing | python-docx | Teams DOCX export format (not VTT) |
| LLM enrichment | Ollama `qwen2.5:7b` | Local, free, sufficient for summarization |
| Process mining export | pandas (XES-compatible CSV) | Ready for PM4Py import |
| Terminal UI | rich | Progress bars, structured output |

---

## Known Limitations

These are real limitations discovered during the POC, not theoretical concerns.

**OCR screen title quality is the main bottleneck.**
F8 Wise renders window titles with icons and partial text that OCR misreads (`×`, `MM`, `0`, `▶·×C`). Around 40% of screen names fall back to "Unknown Screen". This is ERP-specific — other systems may OCR better or worse. The event log is still useful because timestamps, form field contents, and confirmed events are unaffected.

**CLIP zero-shot confidence is weak on enterprise ERP screens (0.35–0.50).**
CLIP was trained on consumer/web imagery. Dense enterprise UI with small text, gray themes, and IFS-style layouts doesn't match its training distribution well. The pipeline uses OCR title as primary identifier and falls back to CLIP only when OCR confidence is low.

**Dense list/table screens are slow on CPU.**
Screens with 700+ OCR regions (invoice list views) take ~6 minutes each on CPU. The pipeline detects these automatically (`region_count > 150` → `layout_type = list_table`) and skips label/value pairing for them, but processing time still accumulates. A GPU cuts this to seconds.

**Toolbar noise.**
ERP toolbars repeat the same buttons (Tasks, Attachments, Output, Help, System Info) on every screen. These are filtered in `event_builder.py` but truncated OCR readings of toolbar text (`evler`, `örevler`, `ptal Et`) may still appear in edge cases.

**Local 7B LLM instruction-following is imperfect.**
Tested with `qwen2.5:7b`. The model occasionally ignores explicit instructions (e.g. "do not recalculate total duration" — it sometimes does anyway). Using a larger model or the Anthropic API produces more reliable structured output.

---

## What Would Make This Significantly Better

In rough priority order:

**1. ERP-specific screen name rules (high impact, low effort)**
A small lookup table mapping known F8/IFS window title patterns to human-readable names would cut "Unknown Screen" occurrences by ~60%. This is maintainable even manually for a specific ERP.

**2. GPU processing (high impact on runtime)**
PaddleOCR and CLIP both run 10–20× faster on CUDA. The 2.5-hour CPU run becomes ~10 minutes.

**3. Multi-keyframe sampling for long scenes**
The POC takes one keyframe per scene. A 17-minute scene (1050 seconds) is represented by one frame. Sampling every N seconds within long scenes would dramatically improve coverage of long navigation sequences.

**4. Stronger LLM (medium impact)**
Claude Sonnet or GPT-4o as the LLM enricher instead of a local 7B model would produce cleaner structured output and better handle noisy screen names in context. The compressed event log is already small enough (~1300 tokens) that API cost is negligible — roughly $0.001 per video session with Haiku.

**5. PM4Py integration**
`audit_trail.csv` is already XES-compatible. Connecting PM4Py's process discovery algorithms (alpha miner, inductive miner) would produce actual BPMN process diagrams directly from the event log.

---

## Project Structure

```
erp-video-miner/
├── main.py                  ← CLI entry point
├── config.yaml              ← all thresholds here, no magic numbers in code
├── requirements.txt
├── pipeline/
│   ├── scene_detector.py    ← PySceneDetect wrapper
│   ├── frame_ocr.py         ← PaddleOCR with Turkish ERP preprocessing
│   ├── clip_classifier.py   ← zero-shot screen classification
│   ├── transcript_parser.py ← Teams DOCX + Whisper fallback
│   ├── event_builder.py     ← state→action inference, confidence scoring
│   ├── llm_enricher.py      ← Ollama / Anthropic, last step only
│   └── models.py            ← shared dataclasses
├── scripts/
│   └── patch_transcript.py  ← update transcript fields without re-running OCR
├── output/                  ← generated, not committed
└── tests/
```

---

## Tests

```bash
pytest tests/
```

Tests use synthetically generated videos (cv2.VideoWriter color-block clips) rather than real ERP footage, so they run without any proprietary data.

---

## Configuration

All tunable parameters are in `config.yaml`:

```yaml
scene_detection:
  threshold: 30.0       # lower = more sensitive to screen changes

ocr:
  lang: "latin"         # not 'tr' — better for mixed Turkish/English ERP UI
  confidence_threshold: 0.65

clip:
  model: "openai/clip-vit-base-patch32"

llm:
  provider: "ollama"    # "ollama" | "anthropic" | "openai"
  model: "qwen2.5:7b"
  max_events_for_llm: 100
```

---

## Why Not Just Use Existing Process Mining Tools?

Commercial task mining tools (UiPath, ABBYY Timeline, Microsoft Process Advisor) record live user interactions in real time. They require agent installation, admin rights, and active sessions.

This project's target scenario is different: **the recording already happened.** The goal is to extract process knowledge from an existing video archive — training sessions, onboarding recordings, support calls — without re-running anything or touching the ERP system.

---

## Status

**This is a proof of concept.** It demonstrates that the approach works. It is not production software.

The pipeline runs end-to-end and produces usable output. The main gap between POC and production is OCR screen name reliability, which is ERP-specific and improvable with relatively little effort.