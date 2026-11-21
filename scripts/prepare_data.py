import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

import logging

logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persian_asr.utils.env import CACHE_ROOT  # noqa: E402  sets HF_HOME first
from persian_asr.config import Config  # noqa: E402

DS_NAME = "kiarashQ/farsi-asr-unified-cleaned"
TEST_HOURS = 2.0
DEV_HOURS = 2.0
BUDGET_HOURS = [10, 25, 50, 100, 200]
DEFAULT_MAX_HOURS = TEST_HOURS + DEV_HOURS + max(BUDGET_HOURS)
MANIFEST_FIELDS = ["id", "file", "text", "duration_sec", "source_path"]


def sanitize_id(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", path).strip("_")[:120]


def write_wav(audio: np.ndarray, sr: int, path: Path) -> None:
    sf.write(str(path), audio.astype(np.float32), sr, format="WAV", subtype="PCM_16")


def append_row_csv(row: dict, path: Path) -> None:
    header = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not header:
            w.writeheader()
        w.writerow(row)


def load_partial(path: Path) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                # csv.DictReader returns every field as a string; downstream
                # np.cumsum over duration_sec requires numbers, else the whole
                # split build crashes with _UFuncNoLoopError on resume runs.
                r["duration_sec"] = float(r["duration_sec"])
                seen[r["id"]] = r
    return seen


def write_status(splits_dir: Path, done_hours: float, total_hours: float) -> None:
    with open(splits_dir / "prepare_status.txt", "w", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} done {done_hours:.2f}h / {total_hours:.2f}h\n")


def materialize_pool(cfg, max_hours: float) -> list[dict]:
    pool_dir = Path(cfg.get("data.processed_dir")) / "pool"
    processed_dir = Path(cfg.get("data.processed_dir"))
    splits_dir = Path(cfg.get("data.splits_dir"))
    pool_dir.mkdir(parents=True, exist_ok=True)
    target_sec = max_hours * 3600

    partial_path = splits_dir / "pool_manifest.partial.csv"
    seen = load_partial(partial_path)
    total_sec = sum(float(r["duration_sec"]) for r in seen.values())
    print(f"Resuming with {len(seen)} utt ({total_sec/3600:.2f}h) already on disk.", flush=True)

    fetch_pool = ThreadPoolExecutor(max_workers=1)

    def fetch(sample_iter):
        try:
            return next(sample_iter)
        except StopIteration:
            return "STOP"
        except (ConnectionError, OSError) as e:
            return ("NETERR", e)

    ds_iter = None

    def new_iter():
        return iter(load_dataset(DS_NAME, split="train", streaming=True))

    pbar = tqdm(total=target_sec / 3600, initial=total_sec / 3600, unit="h", desc="Downloading")
    new_count = 0
    iter_failures = 0
    while total_sec < target_sec:
        if ds_iter is None:
            try:
                ds_iter = new_iter()
                iter_failures = 0
            except Exception as e:
                iter_failures += 1
                print(f"\nIterator init failed ({iter_failures}x): {type(e).__name__}", flush=True)
                if iter_failures >= 12:
                    print("Persistent hub outage → aborting (rerun to resume).", flush=True)
                    sys.exit(3)
                time.sleep(10)
                continue
        try:
            fut = fetch_pool.submit(fetch, ds_iter)
            result = fut.result(timeout=180)
        except TimeoutError:
            print("\nStall watchdog: no row in 180s → abandoning iterator.", flush=True)
            ds_iter = None
            continue

        if result == "STOP":
            print("\nDataset exhausted before reaching target hours.", flush=True)
            break
        if isinstance(result, tuple) and result[0] == "NETERR":
            print(f"\nNetwork error: {result[1]}", flush=True)
            time.sleep(5)
            ds_iter = None
            continue

        row = result
        uid = sanitize_id(row["path"])
        if uid in seen:
            continue

        audio = np.asarray(row["audio"]["array"], dtype=np.float32)
        sr = row["audio"]["sampling_rate"]
        duration = len(audio) / sr
        if duration < 0.5 or duration > 30.0:
            continue

        sub = uid[:2] if uid else "zz"
        wav_path = pool_dir / sub / f"{uid}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(audio, sr, wav_path)

        entry = {
            "id": uid,
            "file": str(wav_path.relative_to(processed_dir)),
            "text": row["text"],
            "duration_sec": round(duration, 3),
            "source_path": row["path"],
        }
        append_row_csv(entry, partial_path)
        seen[uid] = entry
        total_sec += duration
        new_count += 1
        pbar.update(duration / 3600)
        if new_count % 200 == 0:
            write_status(splits_dir, total_sec / 3600, max_hours)

    pbar.close()
    fetch_pool.shutdown(wait=False)
    write_status(splits_dir, total_sec / 3600, max_hours)
    print(f"Materialized {len(seen)} utterances ({total_sec/3600:.2f}h), +{new_count} new.", flush=True)
    return [seen[k] for k in sorted(seen.keys())]


def write_manifest_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  (skip) {path.name}: empty")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def build_splits(manifest: list[dict], cfg) -> None:
    out_dir = Path(cfg.get("data.splits_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cum = np.cumsum([r["duration_sec"] for r in manifest])

    test_end = int(np.searchsorted(cum, TEST_HOURS * 3600)) + 1
    dev_end = int(np.searchsorted(cum, (TEST_HOURS + DEV_HOURS) * 3600)) + 1

    test_rows = manifest[:test_end]
    dev_rows = manifest[test_end:dev_end]
    pool_rows = manifest[dev_end:] if dev_end < len(manifest) else []

    write_manifest_csv(test_rows, out_dir / "test.csv")
    write_manifest_csv(dev_rows, out_dir / "dev.csv")
    print(f"Splits — test: {len(test_rows)} utt, dev: {len(dev_rows)} utt, pool: {len(pool_rows)} utt", flush=True)

    pool_cum = np.cumsum([r["duration_sec"] for r in pool_rows])
    for h in BUDGET_HOURS:
        idx = int(np.searchsorted(pool_cum, h * 3600)) + 1
        budget_rows = pool_rows[:idx]
        if not budget_rows:
            continue
        write_manifest_csv(budget_rows, out_dir / f"budget_{h}h.csv")
        actual = sum(r["duration_sec"] for r in budget_rows) / 3600
        print(f"  budget_{h}h.csv → {len(budget_rows)} utt, {actual:.2f}h", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & split Farsi ASR placeholder data.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_hours", type=float, default=DEFAULT_MAX_HOURS)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    manifest = materialize_pool(cfg, args.max_hours)
    manifest_path = Path(cfg.get("data.splits_dir")) / "pool_manifest.csv"
    write_manifest_csv(manifest, manifest_path)
    build_splits(manifest, cfg)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()