import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

from persian_asr.config import Config
from persian_asr.data.splits import nested_budget_splits
from persian_asr.evaluate.metrics import compute_metrics
from persian_asr.train.finetune import (
    WhisperDataCollator,
    _autocast_context,
    load_audio,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MANIFEST_FIELDS = ["id", "file", "text", "duration_sec", "source_path"]


def load_manifest(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


class WhisperTrainer:
    """Thin wrapper around the shared finetune primitives sized for 16GB cards."""

    def __init__(self, model_name: str = "openai/whisper-large-v3", device: str = None, use_lora: bool = True, cfg: dict = None):
        from peft import LoraConfig, get_peft_model

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg or {}
        self.model_name = model_name

        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.enable_input_require_grads()  # must precede .to(device)
        model = model.to(self.device)

        tokenizer = WhisperTokenizer.from_pretrained(model_name, language="fa")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        if use_lora and self.device == "cuda":
            lora_config = LoraConfig(
                r=int(self.cfg.get("training.lora_rank", 32)),
                lora_alpha=int(self.cfg.get("training.lora_alpha", 64)),
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            for name, p in model.named_parameters():
                p.data = p.data.to(torch.float32 if p.requires_grad else torch.bfloat16)

        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.collator = WhisperDataCollator(feature_extractor, tokenizer)

    def _optimizer(self, lr: float):
        if torch.cuda.is_available() and self.cfg.get("training.optim", "adamw_bnb_8bit") == "adamw_bnb_8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(
                    [p for p in self.model.parameters() if p.requires_grad], lr=lr
                )
            except ImportError:
                logger.warning("bitsandbytes unavailable; falling back to AdamW")
        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )

    def evaluate(self, manifest_path: str) -> dict:
        return self._evaluate_rows(load_manifest(manifest_path))

    def _evaluate_rows(self, rows: list) -> dict:
        if not rows:
            return {}
        return self._generate_and_score(rows)

    def _generate_and_score(self, rows: list) -> dict:
        batch_size = int(self.cfg.get("training.per_device_eval_batch_size", 4))
        predictions, references = [], []

        # Force language/task: without it, Whisper runs per-batch language
        # detection and raises ValueError on mixed predicted languages
        # (transformers PR #28687 default) — same as finetune.evaluate_whisper.
        language = (self.cfg.get("model") or {}).get("language", "fa") if isinstance(self.cfg, dict) else "fa"
        task = (self.cfg.get("model") or {}).get("task", "transcribe") if isinstance(self.cfg, dict) else "transcribe"

        # eval-time KV cache flip (same rationale as finetune.evaluate_whisper)
        train_use_cache = self.model.config.use_cache
        self.model.config.use_cache = True
        self.model.eval()
        try:
            for i in tqdm(range(0, len(rows), batch_size), desc="Evaluating"):
                batch = rows[i : min(i + batch_size, len(rows))]
                audios = [load_audio(b["file"]) for b in batch]
                inputs = self.feature_extractor(audios, sampling_rate=16000, return_tensors="pt")
                input_features = inputs.input_features.to(self.device)

                with torch.no_grad(), _autocast_context(self.device, self.cfg):
                    generated_ids = self.model.generate(
                        input_features, max_length=224, do_sample=False,
                        language=language, task=task,
                    )
                preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                predictions.extend(preds)
                references.extend([b["text"] for b in batch])
        finally:
            self.model.config.use_cache = train_use_cache

        return compute_metrics(predictions, references)

    def fine_tune(
        self,
        train_manifest_path: str,
        val_manifest_path: str,
        num_epochs: int = 3,
        batch_size: int = None,
        learning_rate: float = 1e-4,
        output_dir: str = "outputs",
        accum_steps: int = 4,
    ) -> str:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_rows = load_manifest(train_manifest_path)
        val_rows = load_manifest(val_manifest_path)
        if batch_size is None:
            batch_size = int(self.cfg.get("training.per_device_train_batch_size", 4))

        train_loader = DataLoader(train_rows, batch_size=batch_size, shuffle=True, collate_fn=self.collator)

        optimizer = self._optimizer(learning_rate)
        num_grad_steps = num_epochs * (len(train_loader) // accum_steps + (1 if len(train_loader) % accum_steps else 0))
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=max(num_grad_steps, 1))

        self.model.train()
        progress = tqdm(range(num_grad_steps))
        step = 0
        for epoch in range(num_epochs):
            optimizer.zero_grad(set_to_none=True)
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with _autocast_context(self.device, self.cfg):
                    loss = self.model(**batch).loss / accum_steps
                loss.backward()
                step += 1
                if step % accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                progress.update(1)
                if step % 500 == 0:
                    self.model.save_pretrained(str(output_dir))
                    self.tokenizer.save_pretrained(str(output_dir))
                if step >= num_grad_steps:
                    break
            if step >= num_grad_steps:
                break

        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"[data_efficiency fine_tune] peak VRAM: {peak:.2f} GiB")

        return str(output_dir)


def run_ctc_arm(cfg, budgets: list[int]) -> list[dict]:
    """CTC/XLS-R arm: delegate to the verified run-9 recipe (finetune.train_ctc).

    CTC at budget 0 is meaningless (no vocab/head to evaluate), so unlike the
    Whisper arm there is no zero-shot row. Config should be the XLS-R one
    (configs/xlsr_300m.yaml); the budget splits used are the frozen
    budget_{h}h.csv post-freeze manifests, same as scripts/train.py --mode ctc.
    """
    from persian_asr.train import finetune

    splits_dir = Path(cfg.get("data.splits_dir", "data/splits"))
    if not (splits_dir / "pool_manifest.csv").exists():
        logger.error(f"Pool manifest not found: {splits_dir / 'pool_manifest.csv'} — run prepare_data.py first")
        return []

    results = []
    for budget in budgets:
        if budget == 0:
            continue
        logger.info(f"[xlsr/CTC] Training + evaluating {budget}h budget...")
        res = finetune.train_ctc(cfg, float(budget))
        if not res:
            logger.warning(f"[xlsr/CTC] no result for {budget}h")
            continue
        results.append({
            "family": "xlsr_ctc",
            "budget_hours": budget,
            "cer": round(float(res.get("cer", 1.0)), 4),
            "wer": round(float(res.get("wer", 1.0)), 4),
        })
        logger.info(f"  [xlsr/CTC] {budget}h: CER={res.get('cer', 1.0):.4f}, WER={res.get('wer', 1.0):.4f}")
    return results


def run_data_efficiency(config_path: str, family_override: str = None) -> None:
    cfg = Config.from_yaml(config_path)
    cfg.ensure_dirs()

    family = family_override or cfg.get("experiment.best_model_family", "whisper")
    budgets = cfg.get("experiment.data_budgets", [0, 5, 10, 25, 50, 100])

    if family == "xlsr":
        results = run_ctc_arm(cfg, budgets)
        _write_metrics(cfg, results)
        return

    splits_dir = Path(cfg.get("data.splits_dir", "data/splits"))
    pool_manifest = splits_dir / "pool_manifest.csv"
    if not pool_manifest.exists():
        logger.error(f"Pool manifest not found: {pool_manifest}")
        logger.info("Run prepare_data.py first")
        return

    test_manifest = load_manifest(str(splits_dir / "test.csv"))
    dev_manifest = load_manifest(str(splits_dir / "dev.csv"))
    pool = load_manifest(str(pool_manifest))

    train_hours = sum(float(u.get("duration_sec", 0)) for u in pool) / 3600
    test_hours = sum(float(u.get("duration_sec", 0)) for u in test_manifest) / 3600
    dev_hours = sum(float(u.get("duration_sec", 0)) for u in dev_manifest) / 3600
    logger.info(f"Total pool: {train_hours:.1f}h, test: {test_hours:.1f}h, dev: {dev_hours:.1f}h")

    budgets = cfg.get("experiment.data_budgets", [0, 5, 10, 25, 50, 100])
    use_lora = bool(cfg.get("model.peft.enabled", True)) and device_is_cuda()
    trainer = WhisperTrainer(
        model_name=cfg.get("model.checkpoint", "openai/whisper-large-v3"),
        use_lora=use_lora,
        cfg=cfg.raw,
    )

    num_epochs = int(cfg.get("training.num_epochs", 3))
    accum = int(cfg.get("training.gradient_accumulation_steps", 4))
    lr = float(cfg.get("training.learning_rate", 1e-4))

    results = []
    for budget in budgets:
        if budget == 0:
            logger.info("Evaluating zero-shot performance...")
            metrics = trainer._evaluate_rows(test_manifest) if test_manifest else {}
            results.append({
                "family": "whisper_lora",
                "budget_hours": 0,
                "cer": round(float(metrics.get("cer", 1.0)), 4),
                "wer": round(float(metrics.get("wer", 1.0)), 4),
            })
            continue

        logger.info(f"Processing {budget}h budget...")
        budget_splits = nested_budget_splits(pool, budgets_hours=[budget], seed=42)
        train_rows = budget_splits.get(f"{budget}h", [])
        if not train_rows:
            logger.warning(f"No data for {budget}h budget, skipping")
            continue

        budget_train_path = splits_dir / f"train_{budget}h.csv"
        write_manifest(train_rows, budget_train_path)

        # dev: prefer the held-out dev split, fall back to a slice of train for monitoring only
        dev_rows = dev_manifest if dev_manifest else train_rows[:50]
        dev_temp_path = splits_dir / "dev_temp.csv"
        write_manifest(dev_rows, dev_temp_path)

        output_dir = Path(cfg.get("output.checkpoint_dir", "outputs/checkpoints")) / f"budget_{budget}h"
        trainer.fine_tune(
            str(budget_train_path),
            str(dev_temp_path),
            num_epochs=num_epochs,
            learning_rate=lr,
            output_dir=str(output_dir),
            accum_steps=accum,
        )

        metrics = trainer._evaluate_rows(test_manifest) if test_manifest else {}
        actual_hours = sum(float(u.get("duration_sec", 0)) for u in train_rows) / 3600
        results.append({
            "family": "whisper_lora",
            "budget_hours": budget,
            "actual_hours": round(actual_hours, 2),
            "cer": round(float(metrics.get("cer", 1.0)), 4),
            "wer": round(float(metrics.get("wer", 1.0)), 4),
        })
        logger.info(f"  Budget {budget}h: CER={metrics.get('cer', 1.0):.4f}, WER={metrics.get('wer', 1.0):.4f}")

    _write_metrics(cfg, results)


def _write_metrics(cfg, results: list[dict]) -> None:
    """Single sink for both arms: data_efficiency_results.json + console table."""
    results_path = Path(cfg.get("output.metrics_dir", "outputs/metrics")) / "data_efficiency_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(), "results": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {results_path}")

    print("\n" + "=" * 72)
    print("Data Efficiency Results:")
    print("=" * 72)
    print(f"{'Family':<14} {'Budget (h)':<12} {'CER':<10} {'WER':<10}")
    print("-" * 72)
    for r in results:
        print(f"{r.get('family', '-'):<14} {r['budget_hours']:<12} {r.get('cer', 1.0):<10.4f} {r.get('wer', 1.0):<10.4f}")
    print("=" * 72)


def device_is_cuda() -> bool:
    return torch.cuda.is_available()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the data-efficiency curve over labeled budgets (0/5/10/25/50/100h)."
    )
    parser.add_argument("--config", default="configs/data_efficiency.yaml")
    parser.add_argument("--family", choices=["whisper", "xlsr"], default=None,
                        help="Override experiment.best_model_family")
    args = parser.parse_args()

    run_data_efficiency(args.config, family_override=args.family)


if __name__ == "__main__":
    main()
