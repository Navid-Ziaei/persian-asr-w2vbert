"""Standalone beam-vs-greedy CTC evaluation on a saved checkpoint (frozen v2 test split).

Usage:
  python scripts/eval_beam.py --ckpt outputs/checkpoints/budget_50h_ctc [--beam_width 16]
Writes outputs/metrics/beam_vs_greedy.json with both decode modes.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor  # noqa: E402

from persian_asr.config import Config  # noqa: E402
from persian_asr.train import finetune  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/budget_50h_ctc")
    ap.add_argument("--beam_width", type=int, default=16)
    args = ap.parse_args()

    device = finetune._pick_device()
    cfg = Config.from_yaml("configs/xlsr_300m.yaml")
    test_data = list(csv.DictReader(open("data/splits/v3/test.csv", newline="", encoding="utf-8")))

    model = Wav2Vec2ForCTC.from_pretrained(args.ckpt, ctc_loss_reduction="mean", pad_token_id=0, attention_dropout=0.1)
    model.config.ctc_loss_reduction = "mean"
    model.config.pad_token_id = 0
    model.config.use_cache = False
    model = model.to(device)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.ckpt)
    vocab = json.load(open(Path(args.ckpt) / "ctc_vocab.json", encoding="utf-8"))
    meta = json.load(open(Path(args.ckpt) / "ctc_meta.json", encoding="utf-8"))
    blank_id = meta["blank_id"]

    out = {"checkpoint": args.ckpt, "beam_width": args.beam_width, "generated": datetime.now().isoformat()}
    for mode in ("greedy", "beam"):
        import time
        t0 = time.time()
        m = finetune.evaluate_ctc(
            model, fe, test_data, device, cfg, vocab=vocab, blank_id=blank_id, decode_mode=mode,
        ) if mode == "beam" else finetune.evaluate_ctc(
            model, fe, test_data, device, cfg, vocab=vocab, blank_id=blank_id,
        )
        m["decode_sec"] = round(time.time() - t0, 1)
        out[mode] = m
        print(mode, json.dumps(m), flush=True)

    out_path = Path("outputs/metrics/beam_vs_greedy.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    main()
