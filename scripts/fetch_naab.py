"""Stream a small, source-diverse sample of the naab Persian corpus (no full download).

naab (SLPL/naab) is 126 train shards of plain text (~1 GB each, one paragraph per line). We read
only the first --mb_per_shard MB of --n_shards shards spread evenly across the 126, so the
sample mixes whatever sources the shards are ordered by. Output feeds scripts/build_lm.py.

Usage:
  python scripts/fetch_naab.py --out data/lm_raw/naab_sample.txt --n_shards 12 --mb_per_shard 60
  python scripts/build_lm.py --text data/lm_raw/naab_sample.txt --out outputs/lm/fa --max_lines 4000000

LICENCE: the dataset script says MIT but its paper citation says CC BY-NC-SA 4.0 -- treat as
non-commercial/share-alike until you have checked https://huggingface.co/datasets/SLPL/naab.
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

N_TRAIN_SHARDS = 126
BASE = "https://huggingface.co/datasets/SLPL/naab/resolve/main/data/train-{:05d}-of-{:05d}.txt"


def shard_ids(n_shards: int, offset: int = 0) -> list:
    """n evenly spaced shard indices in [0, 126)."""
    n = max(1, min(n_shards, N_TRAIN_SHARDS))
    return [(int(i * N_TRAIN_SHARDS / n) + offset) % N_TRAIN_SHARDS for i in range(n)]


def stream_shard(idx: int, max_bytes: int, out, retries: int = 3) -> tuple:
    """Copy whole lines from the head of one shard; returns (bytes, lines)."""
    url = BASE.format(idx, N_TRAIN_SHARDS)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "persian-asr-lm-sampler"})
            got = lines = 0
            with urllib.request.urlopen(req, timeout=60) as r:
                for raw in r:
                    got += len(raw)
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if line:
                        out.write(line + "\n")
                        lines += 1
                    if got >= max_bytes:
                        break
            return got, lines
        except Exception as e:  # network hiccup: retry the shard from its start
            print(f"  shard {idx} attempt {attempt + 1} failed: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lm_raw/naab_sample.txt")
    ap.add_argument("--n_shards", type=int, default=12)
    ap.add_argument("--mb_per_shard", type=float, default=60)
    ap.add_argument("--offset", type=int, default=0, help="shift shard choice to get a different sample")
    a = ap.parse_args()

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_b = total_l = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for idx in shard_ids(a.n_shards, a.offset):
            b, l = stream_shard(idx, int(a.mb_per_shard * 1e6), out)
            total_b += b
            total_l += l
            print(f"shard {idx:3d}: {b / 1e6:7.1f} MB, {l} paragraphs", flush=True)
    print(f"done: {total_b / 1e6:.0f} MB, {total_l} paragraphs -> {out_path}")


if __name__ == "__main__":
    main()
