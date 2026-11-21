"""Evaluate a w2v-BERT CTC checkpoint with greedy vs KenLM-fused beam decoding.

alpha/beta are tuned on DEV (grid), then ONE test run uses the best pair.
Usage: python scripts/eval_w2vbert_lm.py --ckpt outputs/checkpoints/budget_5h_w2vbert_ep6 --lm outputs/lm/fa.arpa
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC  # noqa: E402

from persian_asr.decode.lm_fusion import decode_batch, make_decoder  # noqa: E402
from persian_asr.evaluate.metrics import bootstrap_ci, compute_metrics, dump_predictions  # noqa: E402
from persian_asr.train.w2vbert_ctc import greedy_decode, make_collator  # noqa: E402


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def log_probs(model, fe, vocab, data, device, bs=4):
    collate, out = make_collator(fe, vocab), []
    model.eval()
    for i in range(0, len(data), bs):
        b = collate(data[i:i + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            lg = model(input_features=b["input_features"].to(device),
                       attention_mask=b["attention_mask"].to(device)).logits
        lp = torch.log_softmax(lg.float(), -1).cpu().numpy()
        # trim padded frames in proportion to the true (unpadded) input length
        ratio = lp.shape[1] / b["attention_mask"].shape[1]
        for j in range(lp.shape[0]):
            n = max(1, int(round(int(b["attention_mask"][j].sum()) * ratio)))
            out.append(lp[j, :n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lm", required=True, help="KenLM .arpa/.bin")
    ap.add_argument("--splits_dir", default="data/splits/v3")
    ap.add_argument("--dev_n", type=int, default=300)
    ap.add_argument("--alphas", default="0.3,0.5,0.8")
    ap.add_argument("--betas", default="0.0,1.0,2.0")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = json.load(open(Path(a.ckpt) / "vocab.json", encoding="utf-8"))
    inv = {i: c for c, i in vocab.items()}
    fe = SeamlessM4TFeatureExtractor.from_pretrained(a.ckpt)
    model = Wav2Vec2BertForCTC.from_pretrained(a.ckpt).to(device)

    dev = rows(Path(a.splits_dir) / "dev.csv")[:a.dev_n]
    test = rows(Path(a.splits_dir) / "test.csv")
    lp_dev, lp_test = log_probs(model, fe, vocab, dev, device), log_probs(model, fe, vocab, test, device)
    ref_dev, ref_test = [r["text"] for r in dev], [r["text"] for r in test]

    def greedy(lps):
        return [greedy_decode(lp.argmax(-1).tolist(), inv) for lp in lps]

    print("dev greedy", compute_metrics(greedy(lp_dev), ref_dev))
    best = None
    for al, be in itertools.product(map(float, a.alphas.split(",")), map(float, a.betas.split(","))):
        m = compute_metrics(decode_batch(make_decoder(vocab, a.lm, al, be), lp_dev), ref_dev)
        print(f"dev alpha={al} beta={be}", m)
        if best is None or m["cer"] < best[0]:
            best = (m["cer"], al, be)
    _, al, be = best
    hyp_g, hyp_lm = greedy(lp_test), decode_batch(make_decoder(vocab, a.lm, al, be), lp_test)
    res = {"alpha": al, "beta": be, "greedy": compute_metrics(hyp_g, ref_test), "lm": compute_metrics(hyp_lm, ref_test),
           "greedy_ci": bootstrap_ci(hyp_g, ref_test), "lm_ci": bootstrap_ci(hyp_lm, ref_test),
           "ckpt": a.ckpt, "lm_path": a.lm}
    out = Path("outputs/metrics") / f"lm_fusion_{Path(a.ckpt).name}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    dump_predictions(out.with_suffix(".predictions.jsonl"), [r["id"] for r in test], hyp_lm, ref_test)
    print("TEST", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
