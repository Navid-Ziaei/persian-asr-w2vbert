"""w2v-BERT 2.0 CTC fine-tuning (Model C) — trained from the pretrained base.

No community fine-tuned warm start (avoids Common Voice contamination). Char vocab is
built on scoring-normalized TRAIN text only; blank = [PAD]; space = '|'.
"""
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from persian_asr.data.text_normalization import normalize_for_scoring
from persian_asr.evaluate.metrics import compute_metrics, dump_predictions
from persian_asr.utils.audio import load_audio

logger = logging.getLogger(__name__)
SPACE = "|"


def build_vocab(train_texts) -> dict:
    chars = sorted({c for t in train_texts for c in normalize_for_scoring(t).replace(" ", SPACE)})
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for c in chars:
        vocab[c] = len(vocab)
    return vocab


def encode(text: str, vocab: dict) -> list:
    unk = vocab["[UNK]"]
    ids = [vocab.get(c, unk) for c in normalize_for_scoring(text).replace(" ", SPACE)]
    return ids or [unk]


def greedy_decode(ids, vocab_inv: dict, blank: int = 0) -> str:
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            ch = vocab_inv.get(int(i), "")
            out.append(" " if ch == SPACE else ("" if ch == "[UNK]" else ch))
        prev = i
    return " ".join("".join(out).split())


def make_collator(feature_extractor, vocab):
    def collate(rows):
        audios = [np.asarray(load_audio(r["file"]), dtype=np.float32) for r in rows]
        feats = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        out = {"input_features": feats.input_features, "attention_mask": feats.attention_mask}
        labels = [torch.tensor(encode(r["text"], vocab)) for r in rows]
        out["labels"] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return out
    return collate


def load_model(cfg, vocab):
    from transformers import Wav2Vec2BertForCTC

    return Wav2Vec2BertForCTC.from_pretrained(
        cfg.get("model.checkpoint", "facebook/w2v-bert-2.0"),
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=float(cfg.get("training.mask_time_prob", 0.05)),
        mask_feature_prob=float(cfg.get("training.mask_feature_prob", 0.05)),
        mask_feature_length=10,
        layerdrop=float(cfg.get("training.layerdrop", 0.0)),
        ctc_loss_reduction="mean",
        ctc_zero_infinity=bool(cfg.get("training.ctc_zero_infinity", True)),
        add_adapter=True,
        pad_token_id=vocab["[PAD]"],
        vocab_size=len(vocab),
    )


def _autocast(device, cfg):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                          enabled=(device == "cuda" and bool(cfg.get("training.bf16", True))))


@torch.no_grad()
def evaluate(model, feature_extractor, rows, vocab, device, cfg, dump_name="predictions_last_w2vbert.jsonl"):
    inv = {i: c for c, i in vocab.items()}
    bs = int(cfg.get("training.per_device_eval_batch_size", 8))
    collate = make_collator(feature_extractor, vocab)
    model.eval()
    hyps, refs = [], []
    for i in tqdm(range(0, len(rows), bs), desc="w2v-BERT eval"):
        batch_rows = rows[i : i + bs]
        b = collate(batch_rows)
        with _autocast(device, cfg):
            logits = model(input_features=b["input_features"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
        hyps.extend(greedy_decode(ids, inv) for ids in logits.argmax(-1).cpu().tolist())
        refs.extend(r["text"] for r in batch_rows)
    dump_predictions(Path(cfg.get("output.metrics_dir", "outputs/metrics")) / dump_name,
                     [r.get("id") for r in rows], hyps, refs)
    return compute_metrics(hyps, refs)


def train_w2vbert_ctc(cfg, budget_hours: float) -> dict:
    from transformers import SeamlessM4TFeatureExtractor, get_scheduler
    from persian_asr.train.finetune import _build_optimizer, _pick_device, load_whisper_dataset, resolve_splits_dir

    device = _pick_device()
    splits_dir = resolve_splits_dir(cfg)
    data = load_whisper_dataset(str(Path(splits_dir) / "pool_manifest.csv"), splits_dir)
    train = data.get(f"budget_{int(budget_hours)}h", [])
    if not train:
        logger.warning(f"No training data for {budget_hours}h in {splits_dir}")
        return {}
    dev = data.get("validation", [])[:200]
    test = data.get("test", [])

    vocab = build_vocab(r["text"] for r in train)  # TRAIN only: test-only chars -> [UNK]
    fe = SeamlessM4TFeatureExtractor.from_pretrained(cfg.get("model.checkpoint", "facebook/w2v-bert-2.0"))
    model = load_model(cfg, vocab)
    if cfg.get("training.gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    model.to(device)

    bs = int(cfg.get("training.per_device_train_batch_size", 8))
    accum = int(cfg.get("training.gradient_accumulation_steps", 4))
    epochs = int(cfg.get("training.num_epochs", 5))
    lr = float(cfg.get("training.learning_rate", 5e-5))
    loader = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=make_collator(fe, vocab), num_workers=0)
    total = epochs * math.ceil(len(loader) / accum)
    warm = int(cfg.get("training.warmup_steps", 500))
    opt = _build_optimizer(model, cfg)  # 8-bit AdamW by default (fits 16 GB)
    sched = get_scheduler("linear", optimizer=opt, num_warmup_steps=min(warm, total // 10), num_training_steps=total)

    ckpt = Path(cfg.get("output.checkpoint_dir", "outputs/checkpoints")) / f"budget_{int(budget_hours)}h_w2vbert{'_' + cfg.get('output.run_tag') if cfg.get('output.run_tag') else ''}"
    ckpt.mkdir(parents=True, exist_ok=True)
    seen, step, run, start_ep = 0, 0, 0.0, 0
    state_path = ckpt / "train_state.pt"
    if bool(cfg.get("training.resume", False)) and state_path.exists():
        st = torch.load(state_path, map_location="cpu")
        model.load_state_dict(torch.load(ckpt / "model_state.pt", map_location="cpu"))
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start_ep, step, seen = st["epoch"], st["step"], st["seen"]
        logger.info(f"Resumed from epoch {start_ep} (step {step}/{total})")
    stop_ep = min(epochs, start_ep + int(cfg.get("training.max_epochs_this_run", epochs)))
    for ep in range(start_ep, stop_ep):
        model.train()
        opt.zero_grad(set_to_none=True)
        for batch in tqdm(loader, desc=f"epoch {ep + 1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast(device, cfg):
                loss = model(**batch).loss
            (loss / accum).backward()
            run += loss.item()
            seen += 1
            if seen % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    logger.info(f"step {step}/{total} loss {run / (25 * accum):.3f}")
                    run = 0.0
        model.save_pretrained(str(ckpt))  # per-epoch checkpoint
        fe.save_pretrained(str(ckpt))
        (ckpt / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save(model.state_dict(), ckpt / "model_state.pt")
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1, "step": step, "seen": seen}, state_path)
        if dev:
            logger.info(f"epoch {ep + 1} dev: {evaluate(model, fe, dev, vocab, device, cfg, 'predictions_dev_w2vbert.jsonl')}")
    if stop_ep < epochs:
        logger.info(f"Segment done at epoch {stop_ep}/{epochs}; rerun with training.resume: true to continue (test eval runs at the end).")
        return {}
    return evaluate(model, fe, test, vocab, device, cfg)
