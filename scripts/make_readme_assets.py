"""Generate docs/results.png and the example-transcription table for the README."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from persian_asr.data.text_normalization import normalize_for_scoring as norm  # noqa: E402

runs = [("1 epoch", "w2vbert_5h_1ep_probe"), ("3 epochs", "w2vbert_5h_3ep"), ("6 epochs", "w2vbert_5h_ep6")]
cer = [json.loads((ROOT / "results" / f"{n}.json").read_text())["cer"] * 100 for _, n in runs]
wer = [json.loads((ROOT / "results" / f"{n}.json").read_text())["wer"] * 100 for _, n in runs]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, vals, title in ((axes[0], cer, "Test CER (%)"), (axes[1], wer, "Test WER (%)")):
    bars = ax.bar([r[0] for r in runs], vals, color=["#9bb7d4", "#5b8bb8", "#1f5e99"])
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.15)
fig.suptitle("w2v-BERT 2.0 + CTC, 5 h of training data (leak-free splits)", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "results.png", dpi=160)

rows = [json.loads(l) for l in (ROOT / "results" / "predictions_w2vbert_5h_6ep.jsonl").read_text(encoding="utf-8").splitlines()]
exact = sum(norm(r["hyp"]) == norm(r["ref"]) for r in rows)
print(f"exact-match utterances: {exact}/{len(rows)} ({100 * exact / len(rows):.1f}%)")
ex = [r for r in rows if norm(r["hyp"]) != norm(r["ref"]) and 5 <= len(norm(r["ref"]).split()) <= 9][:6]
print("| Reference | Model output |\n|---|---|")
for r in ex:
    print(f"| {norm(r['ref'])} | {norm(r['hyp'])} |")
