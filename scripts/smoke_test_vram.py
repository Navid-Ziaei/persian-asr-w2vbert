import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from persian_asr.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MEL_BINS = {  # per checkpoint family; large-v3 differs from the other Whispers
    "openai/whisper-large-v3": 128,
}


def _peak() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _reserved() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_reserved() / (1024 ** 3)


def smoke_whisper_lora(cfg, batch_size: int, steps: int = 3, do_eval: bool = True) -> dict:
    """Shape-identical (random-weight) LoRA probe for any Whisper checkpoint."""
    from peft import LoraConfig, get_peft_model

    name = cfg.get("model.checkpoint", "openai/whisper-large-v3")
    mel = MEL_BINS.get(name, 80)
    device = "cuda"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = WhisperForConditionalGeneration(WhisperConfig.from_pretrained(name))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = model.to(device)

    lora = LoraConfig(
        r=int(cfg.get("model.peft.rank", 32)),
        lora_alpha=int(cfg.get("model.peft.alpha", 64)),
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora)
    for _, p in model.named_parameters():
        p.data = p.data.to(torch.float32 if p.requires_grad else torch.bfloat16)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    x = torch.randn(batch_size, mel, 3000, device=device, dtype=torch.bfloat16)
    # labels bounded in-range (no -100): device-assert-safe synthetic batch
    labels = torch.randint(0, 50000, (batch_size, 224), device="cuda")
    labels[:, 0] = 50258

    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_features=x, labels=labels).loss / 4
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    train_peak, train_res = _peak(), _reserved()
    results = {"train_peak_gib": round(train_peak, 2), "train_reserved_gib": round(train_res, 2)}

    if do_eval:
        model.config.use_cache = True
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model.generate(x, max_length=224, do_sample=False)
        results["eval_peak_gib"] = round(_peak(), 2)
        results["eval_reserved_gib"] = round(_reserved(), 2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cheap VRAM smoke test: shape-identical random-weight model, real LoRA/optimizer/autocast recipe."
    )
    parser.add_argument("--config", default="configs/whisper_large_v3.yaml")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("smoke_test_vram requires a CUDA device")
        sys.exit(1)

    cfg = Config.from_yaml(args.config)
    batch_size = args.batch_size or int(cfg.get("training.per_device_train_batch_size", 4))
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    logger.info(f"Probe: {cfg.get('model.checkpoint')} LoRA r={cfg.get('model.peft.rank', 32)}, "
                f"batch {batch_size}, bf16+grad-ckpt, on {torch.cuda.get_device_name(0)} ({total_gib:.0f} GiB)")

    res = smoke_whisper_lora(cfg, batch_size, do_eval=not args.skip_eval)
    for k, v in res.items():
        logger.info(f"  {k}: {v}")

    worst = max(res.get("train_reserved_gib", 0), res.get("eval_reserved_gib", 0))
    if worst < total_gib * 0.85:
        logger.info(f"PASS: worst reserved {worst:.2f} GiB <= 85% of {total_gib:.0f} GiB")
    elif worst < total_gib:
        logger.warning(f"TIGHT: worst reserved {worst:.2f} GiB vs card {total_gib:.0f} GiB — expect fragmentation risk")
    else:
        logger.error(f"FAIL: worst reserved {worst:.2f} GiB exceeds card {total_gib:.0f} GiB")
        sys.exit(2)


if __name__ == "__main__":
    main()
