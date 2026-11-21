"""Build leakage-free v3 splits (CPU only, no audio read).

Findings that motivated this (v2 audit, 2026-11-19):
  * 100% of test sentences (1558/1558 unique) also occur in the 50 h train budget
    (same sentence, different clip) -> sentence leakage; flat data curves are suspect.
  * budget_100h is not a subset of budget_200h (pool re-sorted); 200h is really 196 h.
  * No speaker id exists in the manifests (source_path only).

v3 keeps the frozen v2 test set as-is (comparability), removes every pool/dev row whose
scoring-normalized sentence appears in test (or dev), then builds NESTED seeded budgets.
Outputs go to data/splits/v3/ ; v2 files are untouched.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from persian_asr.data.splits import nested_budget_splits  # noqa: E402
from persian_asr.data.text_normalization import normalize_for_scoring  # noqa: E402

SPLITS = ROOT / "data" / "splits"
OUT = SPLITS / "v3"
BUDGETS = [5, 10, 25, 50, 100, 150]
MAX_CLIPS = 5
FIELDS = ["id", "file", "text", "duration_sec", "source_path"]


def read(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(rows, p):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def hours(rows):
    return sum(float(r["duration_sec"]) for r in rows) / 3600


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.csv"):
        old.unlink()  # drop stale budgets from earlier runs
    test, dev, pool = read(SPLITS / "test.csv"), read(SPLITS / "dev.csv"), read(SPLITS / "pool_manifest.csv")
    key = lambda r: normalize_for_scoring(r["text"])
    test_txt = {key(r) for r in test}
    dev = [r for r in dev if key(r) not in test_txt]
    held = test_txt | {key(r) for r in dev}
    held_ids = {r["id"] for r in test} | {r["id"] for r in dev}
    train = [r for r in pool if key(r) not in held and r["id"] not in held_ids]
    print(f"pool {len(pool)} -> {len(train)} after removing {len(pool)-len(train)} rows sharing a sentence with test/dev")

    # Cap repeats of one sentence in train so frequent prompts don't dominate.
    cnt, kept = Counter(), []
    for r in sorted(train, key=lambda r: r["id"]):
        k = key(r)
        cnt[k] += 1
        if cnt[k] <= MAX_CLIPS:
            kept.append(r)
    print(f"capped at {MAX_CLIPS} clips/sentence: {len(kept)} rows, {hours(kept):.1f} h, {len(cnt)} unique sentences")

    write(test, OUT / "test.csv")
    write(dev, OUT / "dev.csv")
    write(kept, OUT / "pool.csv")
    write(kept, OUT / "pool_manifest.csv")  # name expected by load_whisper_dataset
    splits = nested_budget_splits(kept, BUDGETS, seed=42)
    prev = set()
    total_h = hours(kept)
    for h in BUDGETS:
        if h > total_h:
            print(f"budget_{h}h skipped: only {total_h:.1f} h available after dedup")
            continue
        rows = splits[f"{h}h"]
        ids = {r["id"] for r in rows}
        assert prev <= ids, "budgets must be nested"
        prev = ids
        write(rows, OUT / f"budget_{h}h.csv")
        print(f"budget_{h}h: {len(rows)} utt, {hours(rows):.2f} h")
    # post-conditions
    assert not ({key(r) for r in kept} & held), "sentence leak"
    print("OK: nested, sentence-disjoint from test/dev. (Speaker-disjointness cannot be verified: no speaker ids.)")


if __name__ == "__main__":
    main()
