import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from persian_asr.config import Config
from persian_asr.train import finetune

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune or evaluate the chosen ASR model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", default="full_finetune", choices=["zero_shot", "full_finetune", "lora", "ctc", "w2vbert_ctc"])
    parser.add_argument("--budget_hours", type=float, default=30)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    model_name = cfg.get("model.checkpoint")
    if not model_name:
        logger.error("config must set model.checkpoint (e.g. configs/whisper_large_v3.yaml or configs/xlsr_300m.yaml)")
        sys.exit(1)

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Mode: {args.mode}, Budget: {args.budget_hours}h")

    if args.mode == "zero_shot":
        result = finetune.train_zero_shot(cfg)
        logger.info(f"Zero-shot results: {result}")
    elif args.mode == "full_finetune":
        result = finetune.train_full_finetune(cfg, args.budget_hours)
        logger.info(f"Full fine-tune results: {result}")
    elif args.mode == "lora":
        result = finetune.train_lora(cfg, args.budget_hours)
        logger.info(f"LoRA results: {result}")
    elif args.mode == "ctc":
        result = finetune.train_ctc(cfg, args.budget_hours)
        logger.info(f"CTC results: {result}")
    elif args.mode == "w2vbert_ctc":
        from persian_asr.train import w2vbert_ctc
        result = w2vbert_ctc.train_w2vbert_ctc(cfg, args.budget_hours)
        logger.info(f"w2v-BERT CTC results: {result}")
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    # Persist metrics for the data-efficiency table (single source of truth to
    # aggregate from, no need to grep console logs later).
    metrics_dir = Path(cfg.get("output.metrics_dir", "outputs/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if result:
        suffix = "" if args.mode == "zero_shot" else f"_{int(args.budget_hours)}h"
        run_tag = cfg.get("output.run_tag")
        if run_tag:
            suffix += f"_{run_tag}"  # sweep/ablation runs must not clobber canonical metrics
        out_path = metrics_dir / f"{args.mode}{suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": args.mode,
                    "budget_hours": args.budget_hours,
                    "config": args.config,
                    "generated": datetime.now().isoformat(),
                    **result,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
