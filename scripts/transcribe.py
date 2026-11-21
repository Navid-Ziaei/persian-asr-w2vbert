"""Transcribe Persian audio files with a fine-tuned w2v-BERT 2.0 CTC checkpoint.

Usage:
  python scripts/transcribe.py --ckpt outputs/checkpoints/budget_5h_w2vbert_ep6 clip1.wav clip2.mp3

The checkpoint directory must contain model weights, preprocessor_config.json and vocab.json
(all written by `scripts/train.py --mode w2vbert_ctc`). Greedy CTC decoding; audio is resampled
to 16 kHz mono automatically.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC  # noqa: E402

from persian_asr.train.w2vbert_ctc import greedy_decode  # noqa: E402
from persian_asr.utils.audio import load_audio  # noqa: E402


def load(ckpt: str, device: str):
    vocab = json.loads((Path(ckpt) / "vocab.json").read_text(encoding="utf-8"))
    fe = SeamlessM4TFeatureExtractor.from_pretrained(ckpt)
    model = Wav2Vec2BertForCTC.from_pretrained(ckpt).to(device).eval()
    return model, fe, {i: c for c, i in vocab.items()}


@torch.no_grad()
def transcribe(model, fe, inv, path: str, device: str) -> str:
    audio = load_audio(path)
    feats = fe(audio, sampling_rate=16000, return_tensors="pt")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = model(input_features=feats.input_features.to(device),
                       attention_mask=feats.attention_mask.to(device)).logits
    return greedy_decode(logits[0].argmax(-1).cpu().tolist(), inv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("audio", nargs="+")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, fe, inv = load(a.ckpt, device)
    for p in a.audio:
        print(f"{p}\t{transcribe(model, fe, inv, p, device)}")


if __name__ == "__main__":
    main()
