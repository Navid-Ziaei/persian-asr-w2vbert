import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from persian_asr.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persian ASR training and evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", default="train", choices=["train", "evaluate", "data_efficiency"])
    parser.add_argument("--model", default="whisper")
    parser.add_argument("--budget_hours", type=float, default=30)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    if args.mode == "train":
        from scripts.train import main as train_main
        train_main()
    elif args.mode == "evaluate":
        from scripts.evaluate import evaluate_checkpoint
        test_manifest = "data/splits/test.csv"
        results = evaluate_checkpoint(args.checkpoint, test_manifest)
        logger.info(f"Results: {results}")
    elif args.mode == "data_efficiency":
        from scripts.data_efficiency import run_data_efficiency
        run_data_efficiency(args.config)


if __name__ == "__main__":
    main()