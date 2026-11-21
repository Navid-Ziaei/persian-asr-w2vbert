"""Standalone Whisper evaluation (default: leak-free v3 splits + punctuation-neutral scorer).

Usage:
  python scripts/eval_whisper.py zero_shot [--config configs/whisper_large_v3_turbo.yaml]
  python scripts/eval_whisper.py lora --ckpt outputs/checkpoints/budget_5h_lora
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor  # noqa: E402

from persian_asr.config import Config  # noqa: E402
from persian_asr.train import finetune  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["zero_shot", "lora"])
    ap.add_argument("--ckpt", default="outputs/checkpoints/budget_5h_lora")
    ap.add_argument("--config", default="configs/whisper_large_v3.yaml")
    ap.add_argument("--splits_dir", default="data/splits/v3")
    args = ap.parse_args()

    device = finetune._pick_device()
    test_data = list(csv.DictReader(open(Path(args.splits_dir) / "test.csv", newline="", encoding="utf-8")))
    wcfg = Config.from_yaml(args.config)
    name = wcfg.get("model.checkpoint", "openai/whisper-large-v3")
    tok = WhisperTokenizer.from_pretrained(name, language="fa")
    fe = WhisperFeatureExtractor.from_pretrained(name)

    if args.mode == "zero_shot":
        model = WhisperForConditionalGeneration.from_pretrained(name)
        model = model.to(device)
        m = finetune.evaluate_whisper(model, tok, fe, test_data, wcfg, device)
        out = f"outputs/metrics/zero_shot_v3_{name.split('/')[-1]}.json"
        del model
    else:
        from peft import PeftModel
        base = WhisperForConditionalGeneration.from_pretrained(name)
        base.config.use_cache = False
        model = PeftModel.from_pretrained(base, args.ckpt)
        model = model.to(device)
        for _, p in model.named_parameters():
            p.data = p.data.to(torch.float32 if p.requires_grad else torch.bfloat16)
        m = finetune.evaluate_whisper(model, tok, fe, test_data, wcfg, device)
        n_steps = None
        meta_path = Path(args.ckpt) / "lora_meta.json"
        if meta_path.exists():
            n_steps = json.load(open(meta_path, encoding="utf-8")).get("train_steps")
        # budget-aware artifact name: budget_25h_lora -> lora_25h.json (default 5h)
        mb = re.search(r"budget_(\d+)h_lora", args.ckpt.replace("\\", "/"))
        budget = int(mb.group(1)) if mb else 5
        out = f"outputs/metrics/lora_{budget}h_v3_{name.split('/')[-1]}.json"
        (Path(out).parent / "lora_last_eval.json").write_text(
            json.dumps({"ckpt": args.ckpt, **m}, indent=2), encoding="utf-8",
        )
        del model
    torch.cuda.empty_cache()
    print("RESULT", json.dumps(m), flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy("outputs/metrics/predictions_last_whisper.jsonl", Path(out).with_suffix(".predictions.jsonl"))
    payload = {
        "mode": args.mode,
        "model": name,
        "checkpoint": args.ckpt if args.mode == "lora" else None,
        "config": args.config,
        "generated": datetime.now().isoformat(),
        **m,
        "source": f"scripts/eval_whisper.py, splits={args.splits_dir}, scorer=normalize_for_scoring",
    }
    Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
