# ERP Video Süreç Madenciliği

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

> **ERP eğitim videolarından — ERP loglarına erişim olmadan — otomatik audit trail ve süreç dokümantasyonu.**

[🇬🇧 English version](README.md)

---

## Problem

ERP implementasyonları saatlerce eğitim kaydı üretir. Bu videoların içinde sistemin gerçekte nasıl kullanıldığına dair tam bir kayıt vardır — hangi ekranlar, hangi iş akışları, hangi sıralamalar. Ama bu bilgiyi manuel olarak çıkarmak yavaş, tutarsız ve neredeyse hiç yapılmaz.

Aynı zamanda, çoğu ERP sisteminden yapısal event verisi *çıkarmak* DBA erişimi, özel log konfigürasyonları veya pahalı süreç madenciliği lisansları gerektirir. Küçük ekiplerin bunların hiçbiri yoktur.

Bu proje farklı bir yol izliyor: **eğitim videosunun kendisini veri kaynağı olarak ele al.**

---

## Nasıl Çalışır

```
Video (MP4)
  │
  ▼  [1] Sahne Tespiti         deterministik · PySceneDetect
  │     ekran değişimlerini tespit eder, keyframe çıkarır
  │
  ▼  [2] OCR                   deterministik · PaddleOCR
  │     ekran başlıklarını, form alanlarını, buton etiketlerini okur
  │
  ▼  [3] CLIP Sınıflandırma    lokal model · API çağrısı yok
  │     zero-shot ekran tipi tanıma
  │
  ▼  [4] Transkript Hizalama   deterministik · python-docx / Whisper
  │     konuşmayı ekran segmentleriyle eşler, aksiyon kelimelerini tespit eder
  │
  ▼  [5] Event Builder         kural bazlı · model yok
  │     state → action inference, confidence skorlama
  │     çıktı: 136 event (confirmed / inferred / uncertain)
  │
  ▼  [6] LLM Enricher          SADECE SON ADIM · Ollama / Anthropic
        girdi: ~120 filtrelenmiş event (~1300 token)
        çıktı: Markdown formatında yapısal süreç özeti
```

**Temel tasarım kuralı:** LLM hiçbir zaman ham frame, ham OCR metni veya video görmez. Yalnızca sıkıştırılmış, filtrelenmiş bir event log alır — oturum başına tipik olarak 2000 token altında.

---

## Çıktılar

| Dosya | Açıklama |
|------|-------------|
| `output/events.json` | Confidence skorlu yapısal süreç event'leri |
| `output/audit_trail.csv` | Düz audit trail (PM4Py / XES uyumlu) |
| `output/ham_eventler.md` | Tüm ham event listesi — LLM olmadan da her zaman üretilir |
| `output/process_summary.md` | LLM tarafından üretilen süreç özeti |
| `output/enriched_scenes.json` | Sahne başına cache'lenmiş OCR + CLIP + transkript verisi |

---

## Hızlı Başlangıç

```bash
git clone https://github.com/yourname/erp-video-miner
cd erp-video-miner
pip install -r requirements.txt

# Transkriptle (Teams DOCX export)
python main.py --video input.mp4 --transcript transcript.docx

# Transkript olmadan (Whisper fallback)
python main.py --video input.mp4

# OCR'ı yeniden işlemeyi atla (enriched_scenes.json zaten varsa)
python main.py --video input.mp4 --transcript transcript.docx --skip-ocr

# LLM olmadan — sadece deterministik pipeline
python main.py --video input.mp4 --no-llm
```

Lokal LLM için (önerilen, ücretsiz):
```bash
# Ollama kur: https://ollama.com/download
ollama pull qwen2.5:7b
ollama serve
```

Tüm threshold ve model seçimleri `config.yaml`'da — kod içinde magic number yok.

---

## POC Sonuçları

Teams DOCX transkriptli **76 dakikalık bir F8 Wise (IFS) muhasebe modülü eğitim videosu** üzerinde test edildi.

| Metrik | Sonuç |
|--------|--------|
| Tespit edilen sahne | 52 |
| Üretilen event | 136 |
| Confirmed event | 49 (%36) |
| Inferred event | 72 (%53) |
| Uncertain event | 15 (%11) |
| Transkript sinyali olan sahne | 27 / 52 |
| Transkript olmayan sahne (sessiz navigasyon) | 25 / 52 |
| LLM girdisi (filtreleme sonrası) | 121 event / ~1300 token |
| Toplam işlem süresi (CPU, GPU yok) | ~2.5 saat* |

*2.5 saatlik süre tek seferlik bir maliyettir. `enriched_scenes.json` cache'lenir — sonraki çalıştırmalar (event yeniden derleme, LLM yeniden çalıştırma, config ayarlama) saniyeler içinde tamamlanır.

**Doğru tespit edilen ERP ekranları:** Gelen Fatura, Muhasebe Kuralları, FEKA İşleme Kontrol Detayları, Finans, Tedarikçi ekranları.

---

## Teknoloji

| Katman | Araç | Neden |
|-------|------|-----|
| Sahne tespiti | PySceneDetect | UI ekran değişimleri için ayarlanmış ContentDetector |
| OCR | PaddleOCR (`lang='latin'`) | Karışık Türkçe/İngilizce ERP UI metni için `lang='tr'`'den daha iyi |
| Ekran sınıflandırma | CLIP `vit-base-patch32` | Zero-shot — training data gerekmez |
| Transkripsiyon (fallback) | Whisper `small` | `base`'e göre daha iyi Türkçe doğruluğu; word-level timestamp |
| Transkript parse | python-docx | Teams DOCX export formatı (VTT değil) |
| LLM enrichment | Ollama `qwen2.5:7b` | Lokal, ücretsiz, özetleme için yeterli |
| Süreç madenciliği export | pandas (XES uyumlu CSV) | PM4Py import'a hazır |
| Terminal UI | rich | Progress bar, yapısal çıktı |

---

## Bilinen Kısıtlar

Bunlar POC sırasında keşfedilen gerçek kısıtlardır, teorik kaygılar değil.

**OCR ekran başlığı kalitesi ana darboğaz.**
F8 Wise pencere başlıklarını ikonlarla ve OCR'ın yanlış okuduğu kısmi metinlerle render ediyor (`×`, `MM`, `0`, `▶·×C`). Ekran isimlerinin yaklaşık %40'ı "Unknown Screen"e düşüyor. Bu ERP'ye özel — diğer sistemler daha iyi ya da daha kötü OCR sonucu verebilir. Event log yine de kullanışlı çünkü timestamp'ler, form alan içerikleri ve confirmed event'ler bundan etkilenmiyor.

**CLIP zero-shot confidence enterprise ERP ekranlarında zayıf (0.35–0.50).**
CLIP, tüketici/web görselleri üzerinde eğitildi. Küçük metinli, gri temalı, IFS-stili yoğun enterprise UI, training dağılımıyla iyi eşleşmiyor. Pipeline, OCR başlığını birincil tanımlayıcı olarak kullanıyor ve CLIP'e sadece OCR confidence düşük olduğunda başvuruyor.

**Yoğun liste/tablo ekranları CPU'da yavaş.**
700+ OCR bölgesi olan ekranlar (fatura liste görünümleri) CPU'da ~6 dakika sürüyor. Pipeline bunları otomatik tespit ediyor (`region_count > 150` → `layout_type = list_table`) ve bunlar için label/value eşleştirmeyi atlıyor, ama işlem süresi yine de birikiyor. GPU bunu saniyelere indirir.

**Toolbar gürültüsü.**
ERP toolbar'ları her ekranda aynı butonları tekrarlıyor (Görevler, Ekler, Çıktı, Yardım, Sistem Bilgisi). Bunlar `event_builder.py`'da filtreleniyor ama toolbar metninin kesik OCR okumaları (`evler`, `örevler`, `ptal Et`) bazı edge case'lerde hâlâ görünebiliyor.

**Lokal 7B LLM'in talimat takibi kusurlu.**
`qwen2.5:7b` ile test edildi. Model bazen açık talimatları görmezden geliyor (örn. "toplam süreyi yeniden hesaplama" — bazen yine de hesaplıyor). Daha büyük bir model veya Anthropic API'si daha güvenilir yapısal çıktı üretir.

---

## Bunu Önemli Ölçüde Daha İyi Yapacak Şeyler

Kabaca öncelik sırasıyla:

**1. ERP'ye özel ekran adı kuralları (yüksek etki, düşük efor)**
Bilinen F8/IFS pencere başlığı pattern'lerini insan tarafından okunabilir adlara eşleyen küçük bir lookup table, "Unknown Screen" oluşumlarını ~%60 azaltır. Belirli bir ERP için manuel olarak bile sürdürülebilir.

**2. GPU işleme (runtime üzerinde yüksek etki)**
PaddleOCR ve CLIP, CUDA'da 10–20× daha hızlı çalışır. 2.5 saatlik CPU çalışması ~10 dakikaya iner.

**3. Uzun sahneler için multi-keyframe örnekleme**
POC, sahne başına bir keyframe alıyor. 17 dakikalık bir sahne (1050 saniye) tek bir frame ile temsil ediliyor. Uzun sahneler içinde her N saniyede örnekleme, uzun navigasyon sıralarının kapsamını önemli ölçüde iyileştirir.

**4. Daha güçlü LLM (orta etki)**
LLM enricher olarak lokal 7B model yerine Claude Sonnet veya GPT-4o, daha temiz yapısal çıktı üretir ve gürültülü ekran adlarını bağlam içinde daha iyi ele alır. Sıkıştırılmış event log zaten yeterince küçük (~1300 token) olduğundan API maliyeti ihmal edilebilir düzeyde — Haiku ile video oturumu başına kabaca $0.001.

**5. PM4Py entegrasyonu**
`audit_trail.csv` zaten XES uyumlu. PM4Py'nin süreç keşif algoritmalarını (alpha miner, inductive miner) bağlamak, event log'dan doğrudan gerçek BPMN süreç diyagramları üretir.

---

## Proje Yapısı

```
erp-video-miner/
├── main.py                  ← CLI entry point
├── config.yaml              ← tüm threshold'lar burada, kod içinde magic number yok
├── requirements.txt
├── pipeline/
│   ├── scene_detector.py    ← PySceneDetect wrapper
│   ├── frame_ocr.py         ← Türkçe ERP preprocessing ile PaddleOCR
│   ├── clip_classifier.py   ← zero-shot ekran sınıflandırma
│   ├── transcript_parser.py ← Teams DOCX + Whisper fallback
│   ├── event_builder.py     ← state→action inference, confidence skorlama
│   ├── llm_enricher.py      ← Ollama / Anthropic, sadece son adım
│   └── models.py            ← paylaşılan dataclass'lar
├── scripts/
│   └── patch_transcript.py  ← OCR'ı yeniden çalıştırmadan transkript alanlarını güncelle
├── output/                  ← üretilir, commit edilmez
└── tests/
```

---

## Testler

```bash
pytest tests/
```

Testler gerçek ERP görüntüsü yerine sentetik olarak üretilen videolar kullanır (cv2.VideoWriter renk-blok klipleri), böylece proprietary veri olmadan çalışırlar.

---

## Konfigürasyon

Tüm ayarlanabilir parametreler `config.yaml`'da:

```yaml
scene_detection:
  threshold: 30.0       # düşür = ekran değişimlerine daha hassas

ocr:
  lang: "latin"         # 'tr' değil — karışık Türkçe/İngilizce ERP UI için daha iyi
  confidence_threshold: 0.65

clip:
  model: "openai/clip-vit-base-patch32"

llm:
  provider: "ollama"    # "ollama" | "anthropic" | "openai"
  model: "qwen2.5:7b"
  max_events_for_llm: 100
```

---

## Neden Mevcut Süreç Madenciliği Araçlarını Kullanmıyoruz?

Ticari task mining araçları (UiPath, ABBYY Timeline, Microsoft Process Advisor) canlı kullanıcı etkileşimlerini gerçek zamanlı kaydeder. Agent kurulumu, admin yetkisi ve aktif oturumlar gerektirirler.

Bu projenin hedef senaryosu farklı: **kayıt zaten gerçekleşmiş.** Hedef, mevcut bir video arşivinden — eğitim oturumları, onboarding kayıtları, destek çağrıları — hiçbir şeyi yeniden çalıştırmadan veya ERP sistemine dokunmadan süreç bilgisi çıkarmak.

---

## Durum

**Bu bir proof of concept'tir.** Yaklaşımın çalıştığını gösterir. Production yazılımı değildir.

Pipeline baştan sona çalışıyor ve kullanılabilir çıktı üretiyor. POC ile production arasındaki ana fark OCR ekran adı güvenilirliğidir, bu ERP'ye özeldir ve nispeten az efor ile iyileştirilebilir.

---

## Lisans

MIT
