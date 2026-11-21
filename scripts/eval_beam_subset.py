"""Quick probe: greedy vs beam decode on a small RANDOM subset of the frozen v2
test set, for a saved CTC checkpoint. Verifies the fixed decode_ctc_beam on real
model logits end-to-end (subset CERs, sample texts, decode times).

Usage:
  python scripts/eval_beam_subset.py [--ckpt outputs/checkpoints/budget_50h_ctc] [--n 40] [--seed 7]
"""
import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor  # noqa: E402

from persian_asr.config import Config  # noqa: E402
from persian_asr.train import finetune  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/budget_50h_ctc")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--beam_width", type=int, default=16)
    args = ap.parse_args()

    rows = list(csv.DictReader(open("data/splits/test.csv", newline="", encoding="utf-8")))
    rng = random.Random(args.seed)
    subset_nsamples = min(args.n, len(rows))
    subset = rng.sample(rows, subset_nsamples)

    device = finetune._pick_device()
    cfg = Config.from_yaml("configs/xlsr_300m.yaml")
    model = Wav2Vec2ForCTC.from_pretrained(args.ckpt, ctc_loss_reduction="mean", pad_token_id=0, attention_dropout=0.1)
    model.config.ctc_loss_reduction = "mean"
    model.config.pad_token_id = 0
    model.config.use_cache = False
    model = model.to(device)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.ckpt)
    vocab = json.load(open(Path(args.ckpt) / "ctc_vocab.json", encoding="utf-8"))
    blank_id = json.load(open(Path(args.ckpt) / "ctc_meta.json", encoding="utf-8"))["blank_id"]

    import time
    out = {"ckpt": args.ckpt, "n": subset_nsamples, "seed": args.seed, "beam_width": args.beam_width,
           "generated": datetime.now().isoformat()}
    for mode in ("greedy", "beam"):
        t0 = time.time()
        m = finetune.evaluate_ctc(model, fe, subset, device, cfg, vocab=vocab,
                                  blank_id=blank_id, decode_mode=mode)
        m["decode_sec"] = round(time.time() - t0, 1)
        out[mode] = m
        print(mode, json.dumps(m, ensure_ascii=False), flush=True)

    out_path = Path("outputs/metrics/beam_subset_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    main()
