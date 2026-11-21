import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

from persian_asr.config import Config
from persian_asr.train.finetune import load_whisper_dataset, evaluate_whisper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Whisper checkpoint on the held-out set.")
    parser.add_argument("--config", default="configs/whisper_large_v3.yaml")
    parser.add_argument("--checkpoint", required=True, help="Model dir (from train.py) or any HF Whisper checkpoint.")
    parser.add_argument("--test_manifest", default="data/splits/v3/test.csv")
    parser.add_argument("--splits_dir", default="data/splits/v3")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading checkpoint: {args.checkpoint} (device={device})")

    model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint).to(device)
    tokenizer = WhisperTokenizer.from_pretrained(args.checkpoint, language="fa")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.checkpoint)

    if args.test_manifest and Path(args.test_manifest).exists():
        import csv
        with open(args.test_manifest, newline="", encoding="utf-8") as f:
            test_data = list(csv.DictReader(f))
    else:
        dataset = load_whisper_dataset(
            str(Path(args.splits_dir) / "pool_manifest.csv"), args.splits_dir
        )
        test_data = dataset.get("test", [])

    if not test_data:
        logger.error(f"No test data found at {args.test_manifest} (and no test.csv in {args.splits_dir})")
        sys.exit(1)

    metrics = evaluate_whisper(model, tokenizer, feature_extractor, test_data, cfg, device)
    logger.info(f"CER: {metrics.get('cer', float('nan')):.4f}  WER: {metrics.get('wer', float('nan')):.4f}")


if __name__ == "__main__":
    main()
