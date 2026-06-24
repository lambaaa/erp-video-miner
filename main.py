"""
main.py
───────
ERP Video Process Miner — CLI giriş noktası.

Kullanım:
    python main.py --video input.mp4 --transcript input.docx
    python main.py --video input.mp4 --skip-ocr      # enriched_scenes.json'ı yeniden kullan
    python main.py --video input.mp4 --no-llm
    python main.py --video input.mp4 --only-scene-detection
    python main.py --video input.mp4 --only-ocr
"""

from pathlib import Path

import click
import yaml
from rich.console import Console

from pipeline import scene_detector, frame_ocr
from pipeline import transcript_parser, batch_enricher, event_builder, llm_enricher

console = Console()

OUTPUT_DIR_DEFAULT = "./output"
ENRICHED_SCENES_JSON = Path("output") / "enriched_scenes.json"


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """config.yaml dosyasını okur ve dict olarak döner."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--video", required=True, type=click.Path(exists=True), help="MP4/WebM/AVI dosya yolu")
@click.option("--transcript", default=None, type=click.Path(exists=True), help="VTT/SRT/DOCX dosya yolu (opsiyonel)")
@click.option("--output-dir", default=OUTPUT_DIR_DEFAULT, type=click.Path(), help="Çıktı klasörü")
@click.option("--config", "config_path", default="./config.yaml", type=click.Path(), help="Config dosyası")
@click.option("--no-llm", is_flag=True, default=False, help="LLM adımını atla")
@click.option(
    "--skip-ocr", is_flag=True, default=False,
    help="output/enriched_scenes.json zaten varsa OCR+CLIP (Batch Enrichment) adımını atla",
)
@click.option("--only-scene-detection", is_flag=True, default=False, help="Debug: sadece sahne tespiti çalıştır")
@click.option("--only-ocr", is_flag=True, default=False, help="Debug: sadece OCR çalıştır")
@click.option("--verbose", is_flag=True, default=False, help="Detaylı log")
def main(video, transcript, output_dir, config_path, no_llm, skip_ocr, only_scene_detection, only_ocr, verbose):
    cfg = load_config(config_path)
    output_dir = Path(output_dir)

    if str(output_dir) not in (OUTPUT_DIR_DEFAULT, "output"):
        console.print(
            "[yellow][UYARI][/yellow] --output-dir şu an alt modüllere iletilmiyor, "
            "pipeline modülleri sabit 'output/' klasörünü kullanacak."
        )

    if only_scene_detection:
        scenes = scene_detector.detect_scenes(video, cfg["scene_detection"])
        console.print(f"\n[SCENE] {len(scenes)} sahne tespit edildi.")
        for s in scenes[:10]:
            console.print(f"  scene_{s.scene_id:04d} | {s.start_time:6.1f}s-{s.end_time:6.1f}s ({s.duration:5.1f}s)")
        return

    if only_ocr:
        scenes = scene_detector.detect_scenes(video, cfg["scene_detection"])
        ocr_cfg = cfg["ocr"]
        ocr = frame_ocr.TurkishERPOCR(
            use_gpu=ocr_cfg.get("use_gpu", False),
            confidence_threshold=ocr_cfg.get("confidence_threshold", 0.65),
        )
        for s in scenes:
            content = ocr.extract_from_frame(s.keyframe_path)
            console.print(f"scene_{s.scene_id:04d} | title={content.title!r} | buttons={content.buttons}")
        return

    console.print("[1/5] 🎬 Scene Detection")
    scenes = scene_detector.detect_scenes(video, cfg["scene_detection"])
    console.print(f"      → {len(scenes)} sahne tespit edildi.")

    console.print("[2/5] 📝 Transcript Alignment")
    transcript_parser.parse_transcript(transcript, video, scenes, cfg["transcript"])

    if skip_ocr and ENRICHED_SCENES_JSON.exists():
        console.print(f"[3/5] ⏭️  OCR + CLIP — {ENRICHED_SCENES_JSON} mevcut, atlanıyor (--skip-ocr)")
    else:
        if skip_ocr:
            console.print(
                f"[yellow][UYARI][/yellow] --skip-ocr verildi ama {ENRICHED_SCENES_JSON} bulunamadı, "
                "OCR+CLIP normal çalıştırılacak."
            )
        console.print("[3/5] 🔍 OCR + 🎯 CLIP Classification (Batch Enrichment)")
        batch_enricher.run_batch_enrichment(config_path)

    console.print("[4/5] ⚡ Event Builder")
    event_builder.run_event_builder(config_path)

    if not no_llm:
        console.print("[5/5] 🤖 LLM Enrichment")
        llm_enricher.run_llm_enrichment(config_path)
    else:
        console.print("[5/5] ⏭️  LLM Enrichment atlandı (--no-llm)")

    console.print("✅ Tamamlandı!")


if __name__ == "__main__":
    main()
