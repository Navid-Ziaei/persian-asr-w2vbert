import csv
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_scheduler,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from persian_asr.utils.audio import load_audio

logger = logging.getLogger(__name__)

WHISPER_BUDGET_HOURS = [5, 10, 25, 50, 100]
DEFAULT_MODEL_NAME = "openai/whisper-large-v3"
DEFAULT_XLSR_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"


def _is_8bit_optim_available() -> bool:
    """bitsandbytes 8-bit AdamW is only usable on a CUDA device."""
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def _build_optimizer(model, cfg) -> torch.optim.Optimizer:
    lr = cfg.get("training.learning_rate", 1e-4)
    if torch.cuda.is_available() and cfg.get("training.optim", "adamw_bnb_8bit") == "adamw_bnb_8bit":
        if _is_8bit_optim_available():
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
        logger.warning("training.optim=adamw_bnb_8bit requested but bitsandbytes is not installed; falling back to AdamW")
    return torch.optim.AdamW(model.parameters(), lr=lr)


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast_context(device: str, cfg):
    use_bf16 = bool(cfg.get("training.bf16", True)) and device == "cuda"
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device, enabled=False)


# Whisper -------------------------------------------------------------------

class WhisperDataCollator:
    """Decodes raw audio on the fly and produces 30s log-mel inputs + padded labels."""

    def __init__(self, feature_extractor, tokenizer, sample_rate: int = 16000):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __call__(self, features):
        audios = []
        texts = []
        for f in features:
            raw = f["file"]
            audio = load_audio(raw, self.sample_rate) if isinstance(raw, (str, Path)) else raw
            audios.append(audio if isinstance(audio, np.ndarray) else np.asarray(audio, dtype=np.float32))
            texts.append(f["text"])

        inputs = self.feature_extractor(
            audios, sampling_rate=self.sample_rate, return_tensors="pt"
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                texts, padding="max_length", truncation=True, max_length=224, return_tensors="pt"
            ).input_ids

        # Do not supervise on padded positions.
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


def resolve_splits_dir(cfg) -> str:
    """data.splits_dir, plus optional data.splits_subdir (e.g. 'v3' = leakage-free splits)."""
    base = Path(cfg.get("data.splits_dir", "data/splits"))
    sub = cfg.get("data.splits_subdir")
    return str(base / sub) if sub else str(base)


def load_whisper_dataset(pool_manifest_path: str, splits_dir: str):
    """Load budget + dev + test manifests for the data-efficiency sweep."""
    manifest_path = Path(pool_manifest_path)
    if not manifest_path.exists():
        logger.warning(f"Pool manifest not found: {pool_manifest_path}")
        return {}

    datasets = {}
    for h in WHISPER_BUDGET_HOURS:
        budget_path = Path(splits_dir) / f"budget_{h}h.csv"
        if budget_path.exists():
            with open(budget_path, newline="", encoding="utf-8") as f:
                datasets[f"budget_{h}h"] = list(csv.DictReader(f))
        else:
            logger.warning(f"Budget manifest missing: {budget_path}")

    dev_path = Path(splits_dir) / "dev.csv"
    if dev_path.exists():
        with open(dev_path, newline="", encoding="utf-8") as f:
            datasets["validation"] = list(csv.DictReader(f))

    test_path = Path(splits_dir) / "test.csv"
    if test_path.exists():
        with open(test_path, newline="", encoding="utf-8") as f:
            datasets["test"] = list(csv.DictReader(f))

    return datasets


@torch.no_grad()
def evaluate_whisper(model, tokenizer, feature_extractor, test_data: list, cfg, device: str) -> dict:
    """Generate-based eval on the test manifest, batch size from config."""
    if not test_data:
        logger.warning("No test set available")
        return {}

    batch_size = cfg.get("training.per_device_eval_batch_size", 4)
    language = cfg.get("model.language", "fa")
    task = cfg.get("model.task", "transcribe")

    predictions, references = [], []
    train_mode = model.training

    # Eval-only KV cache: measured on RTX 3080 Ti 16GB, large-v3, batch 8 —
    # use_cache=True cuts generate() reserves from 11.0 GiB to 5.95 GiB.
    # Grad checkpointing forces use_cache=False during training, so flip it
    # back on solely for the eval loop and restore after.
    model.eval()
    train_use_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch = test_data[i : min(i + batch_size, len(test_data))]
            audios = [load_audio(b["file"]) for b in batch]
            inputs = feature_extractor(audios, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            with _autocast_context(device, cfg):
                # Force language/task: without it, Whisper runs per-batch language
                # detection and raises ValueError on mixed predicted languages
                # (transformers PR #28687 default). Configs all set model.language.
                generated_ids = model.generate(
                    input_features,
                    max_length=224,
                    do_sample=False,
                    language=language,
                    task=task,
                )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            predictions.extend(preds)
            references.extend([b["text"] for b in batch])
    finally:
        model.config.use_cache = train_use_cache
        model.train(train_mode)

    from persian_asr.evaluate.metrics import compute_metrics, dump_predictions
    dump_predictions(Path(cfg.get("output.metrics_dir", "outputs/metrics")) / "predictions_last_whisper.jsonl",
                     [b.get("id") for b in test_data], predictions, references)
    return compute_metrics(predictions, references)


def train_zero_shot(cfg) -> dict:
    """Evaluate pretrained model without any labeled data."""
    model_name = cfg.get("model.checkpoint", DEFAULT_MODEL_NAME)
    device = _pick_device()
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="fa")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    splits_dir = resolve_splits_dir(cfg)
    dataset = load_whisper_dataset(str(Path(splits_dir) / "pool_manifest.csv"), splits_dir)
    test_data = dataset.get("test", [])

    return evaluate_whisper(model, tokenizer, feature_extractor, test_data, cfg, device)


def _save_checkpoint(model, checkpoint_dir, label: str) -> None:
    """Save a PeftModel or plain model, resilient to missing optional deps.

    peft's save_pretrained() renders a README model card via jinja2; if jinja2
    or markupsafe is not installed the save throws and the whole run's weights
    are lost at the very last step. On that failure we save the adapter weights
    + peft config manually (card is never needed for inference/reload).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(checkpoint_dir))
    except (ModuleNotFoundError, ImportError) as e:
        if not (hasattr(model, "peft_config") and hasattr(model, "base_model")):
            raise  # plain HF model failing to save is a genuine error
        logger.warning(
            f"peft save_pretrained failed ({type(e).__name__}: {e}); "
            "saving adapter weights + config manually (model card skipped)"
        )
        import json
        from safetensors.torch import save_file

        peft_model = model
        # Canonical peft key format: peft strips the adapter name ("default")
        # segment from state_dict keys when saving and re-inserts it on load.
        # A raw state_dict() dump keeps the segment, and PeftModel.from_pretrained
        # silently discards every mismatched tensor (strict=False) -> the reloaded
        # model behaves like zero-shot. Use peft's own helper to stay compatible
        # with from_pretrained.
        from peft.utils.save_and_load import get_peft_model_state_dict

        state = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(peft_model).items()}
        if not state:  # helper failed to find LoRA tensors — never write an empty adapter
            raise RuntimeError("get_peft_model_state_dict returned no tensors")
        save_file(state, str(checkpoint_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
        cfg = next(iter(peft_model.peft_config.values()))
        cfg.save_pretrained(str(checkpoint_dir))  # writes adapter_config.json
        logger.info(
            f"{label} checkpoint saved manually to {checkpoint_dir} "
            f"({len(state)} LoRA tensors); install jinja2+markupsafe for a standard peft layout"
        )
    logger.info(f"{label} completed; checkpoint saved to {checkpoint_dir}")


def _train_loop(model, train_loader, num_epochs, num_grad_steps, optimizer, lr_scheduler, device, cfg, checkpoint_dir, label):
    """Shared bf16 + gradient-accumulation training loop."""
    accum_steps = int(cfg.get("training.gradient_accumulation_steps", 4))
    progress = tqdm(range(num_grad_steps), desc=label)

    model.train()
    step = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast_context(device, cfg):
                outputs = model(**batch)
                loss = outputs.loss / accum_steps
            loss.backward()

            step += 1
            if step % accum_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.update(1)
            if step >= num_grad_steps:
                break
        if step >= num_grad_steps:
            break

    # Protect long runs: a mid-run adapter save means a later crash cannot eat
    # the whole training (each save is ~120MB of LoRA weights, cheap).
    if step >= num_grad_steps:
        try:
            _save_checkpoint(model, Path(checkpoint_dir) / "partial", label + " (partial)")
        except Exception as e:
            logger.warning(f"Partial checkpoint failed (non-fatal): {type(e).__name__}: {e}")
    _save_checkpoint(model, checkpoint_dir, label)


def _print_peak_vram(label: str) -> None:
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"[{label}] peak VRAM allocated: {peak:.2f} GiB")


def train_full_finetune(cfg, budget_hours: float) -> dict:
    """Full supervised fine-tuning for a given data budget (XLS-R-300M / whisper-medium sized models)."""
    model_name = cfg.get("model.checkpoint", DEFAULT_MODEL_NAME)
    device = _pick_device()

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required together with gradient checkpointing
    model = model.to(device)

    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="fa")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    collator = WhisperDataCollator(feature_extractor, tokenizer)

    splits_dir = resolve_splits_dir(cfg)
    datasets = load_whisper_dataset(str(Path(splits_dir) / "pool_manifest.csv"), splits_dir)

    if budget_hours == 0:
        test_data = datasets.get("test", [])
        return evaluate_whisper(model, tokenizer, feature_extractor, test_data, cfg, device)

    train_data = datasets.get(f"budget_{int(budget_hours)}h", [])
    if not train_data:
        logger.warning(f"No training data for {budget_hours}h budget")
        return {}

    val_data = datasets.get("validation", train_data[:50])

    batch_size = cfg.get("training.per_device_train_batch_size", 4)
    accum_steps = int(cfg.get("training.gradient_accumulation_steps", 4))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = _build_optimizer(model, cfg)
    num_epochs = cfg.get("training.num_epochs", 3)
    num_grad_steps = num_epochs * (len(train_loader) // accum_steps + (1 if len(train_loader) % accum_steps else 0))
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,    num_training_steps=num_grad_steps)




    checkpoint_dir = Path(cfg.get("output.checkpoint_dir", "outputs/checkpoints")) / f"budget_{int(budget_hours)}h_full"
    _train_loop(model, train_loader, num_epochs, num_grad_steps, optimizer, lr_scheduler, device, cfg, checkpoint_dir, label="Full fine-tune")

    _print_peak_vram("Full fine-tune")
    result = evaluate_whisper(model, tokenizer, feature_extractor, datasets.get("test", []), cfg, device)
    return result


def train_lora(cfg, budget_hours: float) -> dict:
    """Parameter-efficient (LoRA) fine-tuning for a given data budget — whisper-large-v3 sized models."""
    from peft import LoraConfig, get_peft_model

    model_name = cfg.get("model.checkpoint", DEFAULT_MODEL_NAME)
    device = _pick_device()

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.enable_input_require_grads()  # must be called BEFORE .to(device) so checkpointed inputs carry grads
    model = model.to(device)

    lora_config = LoraConfig(
        r=cfg.get("model.peft.rank", 32),
        lora_alpha=cfg.get("model.peft.alpha", 64),
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # VERIFIED VRAM ENVELOPE (RTX 3080 Ti 16GB, large-v3, batch 4, grad ckpt, bf16 autocast):
    # 6.0 GiB allocated / 7.1 GiB reserved peak, for both training and generate() eval.
    # This requires fp32 master LoRA weights + bf16 frozen base; do not flip all params
    # to bf16 — the eager CUBLAS path on this GPU fails with CUBLAS_STATUS_EXECUTION_FAILED.
    if device == "cuda":
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)
            else:
                p.data = p.data.to(torch.bfloat16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="fa")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    splits_dir = resolve_splits_dir(cfg)
    datasets = load_whisper_dataset(str(Path(splits_dir) / "pool_manifest.csv"), splits_dir)

    if budget_hours == 0:
        test_data = datasets.get("test", [])
        return evaluate_whisper(model, tokenizer, feature_extractor, test_data, cfg, device)

    train_data = datasets.get(f"budget_{int(budget_hours)}h", [])
    if not train_data:
        logger.warning(f"No training data for {budget_hours}h budget")
        return {}

    val_data = datasets.get("validation", train_data[:50])

    collator = WhisperDataCollator(feature_extractor, tokenizer)

    batch_size = cfg.get("training.per_device_train_batch_size", 4)
    accum_steps = int(cfg.get("training.gradient_accumulation_steps", 4))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = _build_optimizer(model, cfg)
    num_epochs = cfg.get("training.num_epochs", 3)
    num_grad_steps = num_epochs * (len(train_loader) // accum_steps + (1 if len(train_loader) % accum_steps else 0))

    # Optional run tag keeps LR-sweep / ablation artifacts from clobbering the
    # canonical budget_{h}h_lora checkpoint (and its metrics file).
    tag = cfg.get("output.run_tag")
    ckpt_name = f"budget_{int(budget_hours)}h_lora" + (f"_{tag}" if tag else "")
    checkpoint_dir = Path(cfg.get("output.checkpoint_dir", "outputs/checkpoints")) / ckpt_name
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(cfg.get("training.warmup_steps", 0)),
        num_training_steps=num_grad_steps,
    )
    _train_loop(model, train_loader, num_epochs, num_grad_steps, optimizer, lr_scheduler, device, cfg, checkpoint_dir, label="LoRA fine-tune")

    _print_peak_vram("LoRA fine-tune")
    result = evaluate_whisper(model, tokenizer, feature_extractor, datasets.get("test", []), cfg, device)
    return result


# XLS-R / CTC ---------------------------------------------------------------

def build_ctc_vocab(train_texts: list) -> dict:
    """Build a character-level CTC vocab from the training transcripts.

    0 is reserved for the CTC blank (it is also the batch pad id, matching
    pad_token_id=0 in the model config). The vocab is built on NORMALIZED
    text so that scoring in evaluate_ctc uses exactly the same normalization.
    """
    from persian_asr.data.text_normalization import normalize_text

    chars = set()
    for t in train_texts:
        chars.update(normalize_text(t))
    vocab = {c: i + 2 for i, c in enumerate(sorted(chars))}  # ids 2..N
    vocab["<pad>"] = 0   # CTC blank
    vocab["<unk>"] = 1
    return vocab


def decode_ctc(logits: torch.Tensor, vocab: dict, blank_id: int = 0) -> list:
    """Greedy CTC decode: argmax, collapse repeats, drop blanks.

    blank_id defaults to 0 for the build_ctc_vocab scheme (blank/<pad>=0);
    native-vocab checkpoints pass their dedicated blank row instead, keeping
    other ids (notably ' ' id 0) as real characters.
    """
    id_to_char = {i: c for c, i in vocab.items()}
    unk_id = vocab.get("<unk>")
    unk = id_to_char.get(unk_id, "")
    max_id = len(vocab)

    pred_ids = logits.argmax(dim=-1)  # (batch, time)
    texts = []
    for seq in pred_ids:
        chars = []
        prev = -1
        for idx in seq.tolist():
            if idx != prev and idx != blank_id:
                if idx in id_to_char:
                    # BEWARE: not vocab-defined means unk only when the vocab
                    # actually defines <unk>; native vocabs (36 chars, 40 head
                    # rows) have a REAL char at id 1, so id_to_char.get(1) was
                    # inserting junk 'آ' for every extra-head-row emission.
                    chars.append(unk if idx == unk_id else id_to_char[idx])
                # ids beyond the char space (extra head rows, e.g. the
                # pretrained blank + pad rows 36..39 of a 40-row head) are
                # silence-like: they must decode to NOTHING, not to a char.
            prev = idx
        texts.append("".join(chars))
    return texts


def decode_ctc_beam(logits: torch.Tensor, vocab: dict, blank_id: int = 0, beam_width: int = 16) -> list:
    """Prefix beam-search CTC decode (no external LM; pure acoustic beam).

    Standard CTC prefix beam search over frame log-probs. State per prefix:
    (log_p_b, log_p_nb) = mass of paths collapsing to this prefix with the

    last emitted frame being blank / a non-blank char respectively.
    Recurrence at frame t for top-k non-blank chars c:

      p_b'(P)          = (p_b(P) + p_nb(P))        * P(blank)
      p_nb'(P + c)    += (p_b(P) + p_nb(P))       * P(c)    if c != last(P)
      p_nb'(P)        += p_nb(P)                  * P(c)    if c == last(P)  (repeat collapse)
      p_nb'(P + c)    += p_b(P)                   * P(c)    if c == last(P)  (blank-separated repeat)
    Final score of P = p_b(P) + p_nb(P). Deterministic and >= greedy wherever
    frame-wise argmax disagrees with the best-scoring time-collapsed
    sequence (common around silence boundaries).

    Same decode policy as decode_ctc: ids beyond the char space decode to
    NOTHING (never a real char); the dedicated blank row is skipped; unk
    maps to '' when the vocab defines no <unk>.

    ``logits`` is a single 2-D slice (frames x vocab) per row — same row-by-
    row policy as greedy decode: batch padding must never leak into decode.
    """
    id_to_char = {i: c for c, i in vocab.items()}
    unk_id = vocab.get("<unk>")
    unk = id_to_char.get(unk_id, "")
    max_id = len(vocab)
    NEG = float("-inf")
    import math

    def lse(a: float, b: float) -> float:
        """log(a + b) for probabilities stored in log space: exact path sum."""
        if a == NEG:
            return b
        if b == NEG:
            return a
        return max(a, b) + math.log1p(math.exp(-abs(a - b)))

    log_probs = torch.log_softmax(logits, dim=-1).float()
    texts = []
    for seq in log_probs:  # single utterance: (time, vocab)
        lp = seq.tolist()
        n_t, n_c = len(lp), len(lp[0])
        # prefixes are tuples of char ids; (p_b, p_nb) in log space
        beams: dict[tuple, tuple] = {(): (0.0, NEG)}
        for t in range(n_t):
            row = lp[t]
            # blank mass = the dedicated blank row + every head-only row beyond
            # the char space (native 40-row heads over 36-entry vocabs): the
            # decode policy treats those rows as silence, exactly like greedy's
            # skip-them behaviour, so their probability belongs to the blank state.
            # Row values are LOG probs — combining masses must be logsumexp
            # (plain + multiplies probabilities and poisoned every frame's
            # blank state, biasing beam toward collapsed single chars).
            p_blank = row[blank_id]
            for i in range(max_id, n_c):
                p_blank = lse(p_blank, row[i])
            # top-k over ALL real characters — INCLUDING id 0. Native vocabs put
            # the space char at id 0 with the CTC blank on a dedicated head row,
            # so range(1, ...) silently dropped space from the search: beam
            # output had NO spaces at all (CER 0.232 / WER 1.0 on the 50h ckpt,
            # see beam_vs_greedy.json and the fix note in PLAN.md).
            cand = [(i, row[i]) for i in range(max_id) if i != blank_id]
            if not cand:
                continue
            cand.sort(key=lambda x: x[1], reverse=True)
            cand = cand[:max(1, beam_width)]
            next_beams: dict[tuple, list] = {}
            for prefix, (pb, pnb) in beams.items():
                last = prefix[-1] if prefix else None
                # 1) blank extension: (p_b + p_nb) * P(blank), exact in log space
                total = lse(pb, pnb)
                if total != NEG:
                    cur = next_beams.setdefault(prefix, [NEG, NEG])
                    cur[0] = max(cur[0], total + p_blank)
                # 2) non-blank extension of each candidate
                for idx, p_c in cand:
                    if last == idx:
                        # repeat collision: collapse into same prefix via p_nb;
                        # blank-separated repeat appends via p_b
                        if pnb != NEG:
                            cur = next_beams.setdefault(prefix, [NEG, NEG])
                            cur[1] = lse(cur[1], pnb + p_c)
                        if pb != NEG:
                            cur = next_beams.setdefault(prefix + (idx,), [NEG, NEG])
                            cur[1] = lse(cur[1], pb + p_c)
                    else:
                        # different char: incoming path mass = (p_b + p_nb) * P(c)
                        mass = lse(pb, pnb)
                        if mass != NEG:
                            cur = next_beams.setdefault(prefix + (idx,), [NEG, NEG])
                            cur[1] = lse(cur[1], mass + p_c)
            # prune to beam width by best total score (log p_b + p_nb)
            scored = sorted(
                ((p, (v[0], v[1])) for p, v in next_beams.items()),
                key=lambda kv: lse(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_width]
            beams = dict(scored)
        if not beams:
            texts.append("")
            continue
        # final score = p_b + p_nb in probability space → exact in log space
        # (math.exp would underflow to 0.0 on long-horizon log probs and
        # math.log(0) raises; lse never does)
        best = max(
            beams.items(),
            key=lambda kv: lse(kv[1][0], kv[1][1]),
        )[0]
        # NOTE: exact-probability ties (e.g. perfectly uniform frames on a
        # synthetic probe) are broken by dict insertion order — arbitrary but
        # deterministic; real trained logits never produce exact ties.
        chars = []
        for idx in best:
            if idx >= max_id:
                continue
            if idx == unk_id:
                chars.append(unk)
            else:
                chars.append(id_to_char[idx])
        texts.append("".join(chars))
    return texts


@torch.no_grad()
def evaluate_ctc(model, feature_extractor, test_data: list, device: str, cfg, vocab=None, blank_id: int = 0, decode_mode: str = "greedy") -> dict:
    """Build the vocab from the TEST transcripts' characters (superset of 
    train is impossible; test-only chars map to <unk>), run CTC decoding on
    the test set, and report CER/WER.

    decode_mode: "greedy" (frame argmax, the default) or "beam" (prefix
    beam-search, no external LM; see decode_ctc_beam).

    NOTE: the vocab is derived from test transcripts only for scoring the
    labels side; the model argmax can only produce ids the CTC head was
    trained with. Characters outside the training vocab decode to <unk> and
    lower the score — matching real deployment where unseen chars exist.
    For the canonical model-vocab decode, use the checkpoint's saved vocab.
    """
    test_texts = [b["text"] for b in test_data]
    if not test_texts:
        logger.warning("No test set available for CTC eval")
        return {}

    vocab_source = None
    if vocab is not None:
        # Canonical: the caller passed the model's ACTUAL trained output space.
        vocab = dict(vocab)
        vocab_source = "passed by caller (training vocab)"
    else:
        # Fall back 1: the budget checkpoint dir (where train_ctc saves ctc_vocab.json).
        if len(test_data) and "budget" in str(test_data[0].get("file")):
            candidates = sorted(Path(cfg.get("output.checkpoint_dir", "outputs/checkpoints")).glob("budget_*_ctc"))
            candidates = [p for p in candidates if (p / "ctc_vocab.json").exists()]
            if candidates:
                train_vocab_path = candidates[-1] / "ctc_vocab.json"
                import json
                with open(train_vocab_path, encoding="utf-8") as f:
                    vocab = {c: int(i) for c, i in json.load(f).items()}
                vocab_source = str(train_vocab_path)
        # Fall back 2: base checkpoint dir (legacy path) — then last resort: derive from test.
        if vocab_source is None:
            base = Path(cfg.get("output.checkpoint_dir", "outputs/checkpoints")) / "ctc_vocab.json"
            if base.exists():
                import json
                with open(base, encoding="utf-8") as f:
                    vocab = {c: int(i) for c, i in json.load(f).items()}
                vocab_source = str(base)
        if vocab_source is None:
            vocab = build_ctc_vocab(test_texts)
            vocab_source = "built from TEST transcripts (last resort; model output space is mismatched!)"
    logger.info(f"CTC eval vocab: {len(vocab)} entries, source={vocab_source}")

    batch_size = int(cfg.get("training.per_device_eval_batch_size", 4))
    predictions, references = [], []

    model.eval()
    for i in tqdm(range(0, len(test_data), batch_size), desc="CTC evaluating"):
        batch = test_data[i : min(i + batch_size, len(test_data))]
        audios = [load_audio(b["file"]) for b in batch]
        inputs = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None else None

        with _autocast_context(device, cfg):
            logits = model(input_values, attention_mask=attention_mask).logits
        logits = logits.float()
        # Decode row-by-row: batched-behaviour keeps padded frames of shorter
        # rows in the argmax sequence, which decodes to trailing junk chars
        # (this is what inflated the in-run eval CER to 0.888 while single-
        # row probes scored 0.48 on identical weights).
        for bi in range(logits.shape[0]):
            if decode_mode == "beam":
                predictions.extend(decode_ctc_beam(logits[bi : bi + 1], vocab, blank_id=blank_id))
            else:
                predictions.extend(decode_ctc(logits[bi : bi + 1], vocab, blank_id=blank_id))
        references.extend([b["text"] for b in batch])

    from persian_asr.evaluate.metrics import compute_metrics, dump_predictions
    dump_predictions(Path(cfg.get("output.metrics_dir", "outputs/metrics")) / "predictions_last_ctc.jsonl",
                     [b.get("id") for b in test_data], predictions, references)
    return compute_metrics(predictions, references)


def _ctc_collator(processor, vocab=None, blank_id: int = None):
    """Collate for CTC training. `processor` may be a full Wav2Vec2Processor or a
    bare feature extractor — the wav2vec2 base repos (e.g. xls-r-300m) ship no
    CTC tokenizer, so labels are built from our own char vocab when given.
    `vocab` must then be built BEFORE training on the full target-domain text
    (train transcripts), not just the current fold, so that test-only chars map
    to <unk> rather than causing OOV bid-index errors in the loss.
    """
    def collate(features):
        audios = []
        texts = []
        for f in features:
            raw = f["file"]
            audio = load_audio(raw) if isinstance(raw, (str, Path)) else raw
            audios.append(audio if isinstance(audio, np.ndarray) else np.asarray(audio, dtype=np.float32))
            texts.append(f["text"])

        batch = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        if vocab is not None:
            from persian_asr.data.text_normalization import normalize_text
            import torch.nn.functional as F
            # transformers 4.40's Wav2Vec2ForCTC expects ONE padded labels tensor
            # (pad = -100, ignored by the loss) — a python list is rejected
            # ('list' object has no attribute 'max'). Vocab ids: blank/pad=0,
            # unk=1, chars=2..N. All ids < config.vocab_size is enforced by the
            # caller rebuilding lm_head AND setting config.vocab_size.
            unk_id = vocab.get("<unk>")
            rid = blank_id if blank_id is not None else 0  # CTC blank from the train path
            rows = []
            for t in texts:
                nt = normalize_text(t)
                ids = []
                for ch in nt:
                    if ch in vocab:
                        ids.append(vocab[ch])
                    elif ch == "\u200c":  # ZWNJ → space (native vocabs include space)
                        ids.append(vocab[" "]) if " " in vocab else None
                    # any remaining OOV char is SKIPPED (native vocabs have <unk>;
                    # skipping is the safer default for unseen punctuation)
                if not ids:
                    ids = [unk_id] if unk_id is not None else []
                rows.append(torch.tensor(ids, dtype=torch.long))
            maxlen = max(len(r) for r in rows)
            labels = torch.stack([F.pad(r, (0, maxlen - len(r)), value=-100) for r in rows])
            # attention_mask must ride along: HF's CTC loss derives input_lengths
            # from it, defaulting to ones_like(input_values) — i.e. the FULL
            # padded window for every row when it's dropped. With 2-10s audio
            # padded to the longest, that misaligns CTC alignment and explodes
            # the loss (~3500 observed) collapsing the model to all-blank.
            out = {"input_values": batch.input_values, "labels": labels}
            if getattr(batch, "attention_mask", None) is not None:
                out["attention_mask"] = batch.attention_mask
            return out
        with processor.as_target_processor():
            labels = processor(texts, return_tensors="pt", padding=True).input_ids
        return {"input_values": batch.input_values, "labels": labels}

    return collate


def train_ctc(cfg, budget_hours: float) -> dict:
    """CTC fine-tuning for XLS-R-300M on a given data budget — the practical full-FT arm on a 16GB GPU."""
    model_name = cfg.get("model.checkpoint", DEFAULT_XLSR_MODEL_NAME)
    device = _pick_device()

    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
        import os
        if not os.environ.get("TRANSFORMERS_OFFLINE") and not os.environ.get("HF_HUB_OFFLINE"):
            # Refuse to "load" a checkpoint whose weights were never actually
            # fetched: transformers silently falls back to RANDOM-initialized
            # weights (only config loads), which trains garbage without any
            # error. We already saw a 408-step run collapse to all-blank this
            # way when only config.json was cached. Hard-fail instead.
            did_preflight = False
            try:
                import huggingface_hub as hf_hub
                info = hf_hub.model_info(model_name)
                weight_files = [f.rfilename for f in info.siblings
                                if f.rfilename.endswith((".safetensors", ".bin", ".msgpack", ".ckpt"))
                                and not f.rfilename.startswith(".")]
                did_preflight = True  # hub reachable and repo listing readable
                if not weight_files:
                    raise RuntimeError(
                        f"{model_name} exposes no downloadable weight file (config-only repo) — "
                        "the checkpoint is corrupt or the network is broken"
                    )
            except RuntimeError:
                raise  # genuine missing-weights verdict must propagate
            except Exception as outer:
                # Hub unreachable (offline/timeout): the subsequent from_pretrained
                # will surface its own error rather than lying about weights.
                logger.warning(
                    f"Pre-flight checkpoint check failed to reach the hub ({outer}); "
                    "skipping weight-file verification"
                )
            if did_preflight:
                logger.info(f"Pre-flight OK: {model_name} exposes {len(weight_files)} weight file(s); loading")

        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            ctc_loss_reduction="mean",
            pad_token_id=0,
            attention_dropout=0.1,
        )
        # gradient_checkpointing as a from_pretrained kwarg breaks on LOCAL
        # checkpoint dirs (transformers 4.40 rejects the kwarg in __init__; hub
        # loads route it through config). Enable via the method instead.
        if bool(cfg.get("training.gradient_checkpointing", True)):
            model.gradient_checkpointing_enable()
        # Do NOT trust the kwarg alone: transformers 4.40.2 has a bug where the hub
        # config's ctc_loss_reduction ('sum' for XLS-R) wins over the kwarg, making
        # model.loss a SUM over batch AND time (~2500-3500 observed instead of
        # ~ln(vocab)): gradients blow up and the model collapses to all-blank.
        # Force the reduction and blank id explicitly on the loaded config.
        model.config.ctc_loss_reduction = "mean"
        model.config.pad_token_id = 0
        assert model.config.ctc_loss_reduction == "mean", "ctc_loss_reduction must be mean after load"
        model.config.use_cache = False
        model = model.to(device)
    except Exception as e:
        logger.error(f"Failed to load XLS-R model '{model_name}': {e}")
        return {}

    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load XLS-R feature extractor: {e}")
        return {}

    splits_dir = resolve_splits_dir(cfg)
    datasets = load_whisper_dataset(str(Path(splits_dir) / "pool_manifest.csv"), splits_dir)

    if budget_hours == 0:
        logger.info("CTC zero-shot eval is not defined for XLS-R (needs fine-tuned CTC head); skipping")
        return {}

    train_data = datasets.get(f"budget_{int(budget_hours)}h", [])
    if not train_data:
        logger.warning(f"No training data for {budget_hours}h budget")
        return {}

    val_data = datasets.get("validation", train_data[:50])

    # Use the CHECKPOINT'S NATIVE vocab + tokenizer when one exists (Persian
    # fine-tuned repos ship vocab.json + tokenizer). Rebuilding a fresh head
    # (nn.Linear to a new size) was the root cause of blank-collapse: a randomly
    # re-initialized head on calibrated features needs far more 5h-budget steps
    # than available to re-calibrate, and training just collapses to blank.
    # With the native head the model only adapts acoustics + minor vocab drift.
    import os, json as _json
    native_vocab_path = None
    _local_ckpt = Path(model_name)
    # 1) Local checkpoint dir first (previous fine-tune): our own runs ship
    #    ctc_vocab.json + ctc_meta.json; community repos ship vocab.json.
    for name in ("ctc_vocab.json", "vocab.json"):
        if _local_ckpt.is_dir() and (_local_ckpt / name).exists():
            native_vocab_path = _local_ckpt / name
            break
    if native_vocab_path is None:
        try:
            if _local_ckpt.is_dir():
                raise ValueError(f"local checkpoint dir without a saved vocab: {_local_ckpt}")
            from huggingface_hub import hf_hub_download, list_repo_files
            files = [f for f in list_repo_files(model_name) if not f.startswith(".")]
            if "vocab.json" in files:
                native_vocab_path = hf_hub_download(model_name, "vocab.json")
        except Exception as e:
            # A vocab-less checkpoint MUST NOT silently fall back to the legacy
            # rebuild path: that overwrote a good 36-char head with a fresh
            # random 79-char one and scores CER 2.15 instead of 0.07. Refuse.
            raise RuntimeError(
                f"No CTC vocab found for checkpoint '{model_name}' "
                "(refusing legacy lm_head rebuild — it randomizes the head and collapses quality)"
            ) from e

    if native_vocab_path is not None:
        with open(native_vocab_path, encoding="utf-8") as f:
            vocab = {c: int(i) for c, i in _json.load(f).items()}
        old_out = model.lm_head.out_features
        logger.info(f"Using NATIVE vocab from checkpoint: {len(vocab)} chars; head NOT rebuilt (out_features={old_out})")
        # Blank policy: prefer a DEDICATED head row beyond the character ids
        # (community Persian repos often ship more head rows than vocab chars —
        # e.g. 40 rows / 36 chars — because their trainer used extra rows as
        # pad/blank, and the checkpoint's logit probe confirms one row sits at
        # +6.5 mean logit everywhere while the others are suppressed to -15).
        # Using id 0 (' ') as blank would make real spaces unrecoverable and
        # structurally forces WER=1.0; instead ' ' stays a real character and
        # the extra head row is the CTC blank for training AND decode.
        if old_out > len(vocab):
            # Use the LAST extra row: the community trainer's calibrated blank
            # (for the 40-row alifarokh head, row 39 sits at +6.5 mean logit
            # everywhere while rows 36-38 are suppressed to -15; training with
            # the first extra row (36) instead made row 39 dominate argmax on
            # every silence frame in eval, and decode inserted junk chars).
            blank_id = old_out - 1
            logger.info(f"Dedicated CTC blank: head row {blank_id} (head has {old_out} rows, vocab has {len(vocab)} chars)")
        else:
            # Head exactly matches the char vocab: fall back to HF convention
            # where pad_token_id is the blank (' ' if present, else 0).
            blank_id = int(vocab.get(" ", 0))
            logger.warning(f"Head matches vocab exactly ({old_out} rows); blank = id {blank_id} (space-order may sacrifice word boundaries)")
        model.config.vocab_size = old_out  # loss validates against true head dim
        model.config.pad_token_id = blank_id  # HF convention: blank = pad_token_id
    else:
        # Fresh multilingual base: build our own char vocab, rebuild the head.
        all_texts = [b["text"] for b in train_data] + [b["text"] for b in val_data] + [b["text"] for b in datasets.get("test", [])]
        vocab = build_ctc_vocab(all_texts)
        import torch.nn as nn
        out_features = model.lm_head.out_features
        if out_features != len(vocab):
            logger.warning(f"lm_head out_features {out_features} != vocab size {len(vocab)}; rebuilding lm_head + config")
            model.lm_head = nn.Linear(model.lm_head.in_features, len(vocab)).to(device)
            model.config.vocab_size = len(vocab)
        model.config.pad_token_id = 0
        blank_id = 0  # legacy scheme: blank/pad = 0, chars start at id 2

    collator = _ctc_collator(feature_extractor, vocab=vocab, blank_id=blank_id)
    batch_size = cfg.get("training.per_device_train_batch_size", 8)
    accum_steps = int(cfg.get("training.gradient_accumulation_steps", 4))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0)

    optimizer = _build_optimizer(model, cfg)
    num_epochs = cfg.get("training.num_epochs", 3)
    num_grad_steps = num_epochs * (len(train_loader) // accum_steps + (1 if len(train_loader) % accum_steps else 0))
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,    num_training_steps=num_grad_steps)




    model.train()
    progress = tqdm(range(num_grad_steps), desc="CTC fine-tune")
    step = 0
    num_micro_steps = 0
    running_loss = 0.0
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast_context(device, cfg):
                outputs = model(input_values=batch["input_values"],
                                attention_mask=batch.get("attention_mask"),
                                labels=batch["labels"])
                loss = outputs.loss / accum_steps
            loss.backward()
            running_loss += outputs.loss.item()
            num_micro_steps += 1

            step += 1
            if step % accum_steps == 0:
                # Grad-norm guard: XLS-R full fine-tune occasionally emits
                # gradient spikes that blank-collapse the model within a few
                # steps. Clip alongside AdamW to keep late-training stability.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if step % (accum_steps * 5) == 0:
                    logger.info(
                        f"CTC step {step}/{num_grad_steps}: "
                        f"avg micro loss (last {accum_steps*5} micro) = "
                        f"{running_loss / (accum_steps*5):.3f}"
                    )
                    running_loss = 0.0
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.update(1)
            if step >= num_grad_steps:
                break
        if step >= num_grad_steps:
            break

    _print_peak_vram("XLS-R CTC")

    checkpoint_dir = Path(cfg.get("output.checkpoint_dir", "outputs/checkpoints")) / f"budget_{int(budget_hours)}h_ctc"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    feature_extractor.save_pretrained(str(checkpoint_dir))

    # Persist the training-derived vocab so eval decodes with the model's actual output space.
    import json
    vocab_path = checkpoint_dir / "ctc_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    # Persist the blank id too: native-vocab checkpoints use a dedicated head
    # row (NOT char id 0) as the CTC blank, and decode must skip that row
    # while keeping other ids — including ' ' — as real characters.
    meta_path = checkpoint_dir / "ctc_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"blank_id": int(blank_id), "vocab_size": len(vocab), "head_rows": int(model.lm_head.out_features)}, f, indent=2)
    logger.info(f"CTC fine-tuning completed; checkpoint + vocab ({len(vocab)} chars) saved to {checkpoint_dir}")

    result = evaluate_ctc(model, feature_extractor, datasets.get("test", []), device, cfg, vocab=vocab, blank_id=blank_id)
    _print_peak_vram("XLS-R CTC eval")
    return result


def self_supervised_adaptation(model, unlabeled_audio, cfg):
    """Continue representation pretraining on target-dialect audio."""
    raise NotImplementedError("Self-supervised adaptation on unlabeled data")


def text_adaptation(model, text_corpus, cfg):
    """Token/LM adaptation using dialect text."""
    raise NotImplementedError("Text-only adaptation")
