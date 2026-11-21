"""Build a Persian n-gram LM corpus (+ ARPA if the KenLM `lmplz` binary exists).

Usage: python scripts/build_lm.py --text corpus.txt [--text more.txt ...] --out outputs/lm/fa
Text is normalized with normalize_for_scoring (same as the scorer / CTC vocab). Any line equal to
a test or dev sentence is REMOVED so the LM cannot leak the evaluation sets. Use EXTERNAL text
(Wikipedia, OSCAR, ...); the training transcripts alone are too small and overlap the domain.
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from persian_asr.data.text_normalization import normalize_for_scoring  # noqa: E402


SENT_SPLIT = re.compile(r"[.!?؟۔؛]+|\s*\n\s*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", required=True)
    ap.add_argument("--splits_dir", default="data/splits/v3")
    ap.add_argument("--out", default="outputs/lm/fa")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--max_lines", type=int, default=0, help="stop after N kept sentences (0 = all)")
    a = ap.parse_args()

    held = set()
    for name in ("test.csv", "dev.csv"):
        with open(Path(a.splits_dir) / name, newline="", encoding="utf-8") as f:
            held |= {normalize_for_scoring(r["text"]) for r in csv.DictReader(f)}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with open(f"{out}.txt", "w", encoding="utf-8") as w:
        for path in a.text:
            for line in open(path, encoding="utf-8"):
                # corpora (naab/OSCAR/Wikipedia) are paragraphs: split into sentences first
                for sent in SENT_SPLIT.split(line):
                    n = normalize_for_scoring(sent)
                    if len(n.split()) < 2:
                        continue
                    if n in held:
                        dropped += 1
                        continue
                    w.write(n + "\n")
                    kept += 1
                if a.max_lines and kept >= a.max_lines:
                    break
            if a.max_lines and kept >= a.max_lines:
                break
    print(f"corpus: {kept} lines, {dropped} held-out sentences removed -> {out}.txt")

    lmplz = shutil.which("lmplz")
    if not lmplz:
        print("KenLM 'lmplz' not found: build the ARPA elsewhere (Linux/WSL):\n"
              f"  lmplz -o {a.order} --prune 0 0 1 < {out}.txt > {out}.arpa")
        return
    with open(f"{out}.txt", "rb") as i, open(f"{out}.arpa", "wb") as o:
        subprocess.run([lmplz, "-o", str(a.order), "--prune", "0", "0", "1"], stdin=i, stdout=o, check=True)
    print(f"ARPA written: {out}.arpa")


if __name__ == "__main__":
    main()
