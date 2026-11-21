from jiwer import cer, wer

from persian_asr.data.text_normalization import normalize_for_scoring


def compute_metrics(predictions, references, language: str = "fa") -> dict:
    """CER/WER on punctuation-free, ZWNJ-as-space text (see normalize_for_scoring).

    Empty references are dropped (jiwer raises on them); empty hypotheses are kept.
    """
    pairs = [(normalize_for_scoring(r), normalize_for_scoring(p)) for p, r in zip(predictions, references)]
    pairs = [(r, p) for r, p in pairs if r]
    refs, preds = [r for r, _ in pairs], [p for _, p in pairs]
    return {"cer": cer(refs, preds), "wer": wer(refs, preds)}


def dump_predictions(path, ids, predictions, references) -> None:
    """Write per-utterance hyp/ref (JSONL) so any run can be re-scored or bootstrapped later."""
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, p, r in zip(ids, predictions, references):
            f.write(json.dumps({"id": i, "hyp": p, "ref": r}, ensure_ascii=False) + "\n")


def bootstrap_ci(predictions, references, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Utterance-level bootstrap CI for corpus CER/WER (differences ~0.0002 are noise)."""
    import numpy as np
    from jiwer import process_characters, process_words

    pairs = [(normalize_for_scoring(r), normalize_for_scoring(p)) for p, r in zip(predictions, references)]
    pairs = [x for x in pairs if x[0]]

    def counts(fn, r, h):
        o = fn(r, h)
        return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits

    c = np.array([counts(process_characters, r, h) for r, h in pairs], dtype=float)
    w = np.array([counts(process_words, r, h) for r, h in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    out = {}
    for name, a in (("cer", c), ("wer", w)):
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        vals = a[idx, 0].sum(1) / a[idx, 1].sum(1)
        out[name] = {"mean": float(a[:, 0].sum() / a[:, 1].sum()),
                     "lo": float(np.quantile(vals, alpha / 2)), "hi": float(np.quantile(vals, 1 - alpha / 2))}
    return out
