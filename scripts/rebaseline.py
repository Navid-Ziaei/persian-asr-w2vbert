"""Re-baseline all model artifacts on the frozen v2 test split.

The original test/dev splits were rewritten during a resume bug (prepare_data
re-sorted ids including new rows, sliding the 4h test+dev window). The pool
ordering was recovered exactly (validated against the surviving budget_5h.csv),
but the old test manifest itself is not byte-recoverable. Standing policy: the
frozen v2 test split is now the canonical evaluation set, and every reported
metric is re-measured on it with the same decode paths as the original runs.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor  # noqa: E402
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor  # noqa: E402

from persian_asr.config import Config  # noqa: E402
from persian_asr.train import finetune  # noqa: E402


def load_test_data():
    import csv
    with open("data/splits/test.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    device = finetune._pick_device()
    test_data = load_test_data()
    results = {}

    # --- XLS-R CTC checkpoints (5/10/25/50h): same decode path as training-time eval
    for h in (5, 10, 25, 50):
        ckpt = Path(f"outputs/checkpoints/budget_{h}h_ctc")
        if not ckpt.exists():
            print(f"skip {h}h (no ckpt)", flush=True)
            continue
        model = Wav2Vec2ForCTC.from_pretrained(str(ckpt), ctc_loss_reduction="mean", pad_token_id=0, attention_dropout=0.1)
        model.config.ctc_loss_reduction = "mean"
        model.config.pad_token_id = 0
        model.config.use_cache = False
        model = model.to(device)
        fe = Wav2Vec2FeatureExtractor.from_pretrained(str(ckpt))
        vocab = json.load(open(ckpt / "ctc_vocab.json", encoding="utf-8"))
        meta = json.load(open(ckpt / "ctc_meta.json", encoding="utf-8"))
        cfg = Config.from_yaml("configs/xlsr_300m.yaml")
        m = finetune.evaluate_ctc(model, fe, test_data, device, cfg, vocab=vocab, blank_id=meta["blank_id"])
        results[f"ctc_{h}h"] = m
        print(f"ctc {h}h: {m}", flush=True)
        del model
        torch.cuda.empty_cache()

    # --- Whisper zero-shot
    wcfg = Config.from_yaml("configs/whisper_large_v3.yaml")
    model_name = wcfg.get("model.checkpoint", "openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    tok = WhisperTokenizer.from_pretrained(model_name, language="fa")
    fe = WhisperFeatureExtractor.from_pretrained(model_name)
    m = finetune.evaluate_whisper(model, tok, fe, test_data, wcfg, device)
    results["whisper_zero_shot"] = m
    print("whisper zero-shot:", m, flush=True)
    del model
    torch.cuda.empty_cache()

    # --- Whisper LoRA (5h): base + PEFT adapter
    try:
        from peft import PeftModel
        base = WhisperForConditionalGeneration.from_pretrained(model_name)
        base.config.use_cache = False
        peft_model = PeftModel.from_pretrained(base, "outputs/checkpoints/budget_5h_lora")
        peft_model = peft_model.to(device)
        if device == "cuda":
            for name, p in peft_model.named_parameters():
                p.data = p.data.to(torch.float32 if p.requires_grad else torch.bfloat16)
        m = finetune.evaluate_whisper(peft_model, tok, fe, test_data, wcfg, device)
        results["whisper_lora_5h"] = m
        print("whisper lora 5h:", m, flush=True)
        del peft_model
        torch.cuda.empty_cache()
    except Exception as e:
        print("lora eval failed:", type(e).__name__, e, flush=True)

    stamp = datetime.now().isoformat()
    out = Path("outputs/metrics/rebaseline_v2.json")
    out.write_text(json.dumps({"generated": stamp, "split": "frozen v2 (test.csv restored from session-1 ids; pool order validated vs budget_5h.csv)", "n_test": len(test_data), "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
