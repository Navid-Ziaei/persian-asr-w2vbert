import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import WhisperFeatureExtractor, WhisperTokenizer

from persian_asr.train import finetune

torch.manual_seed(0)
np.random.seed(0)


@pytest.fixture(scope="module")
def tokenizer_and_fe():
    # tiny public checkpoint, smallest available, only used for tokenizer/FE shapes
    tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="fa")
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    return tok, fe


def test_collator_produces_trainable_batch(tokenizer_and_fe):
    tok, fe = tokenizer_and_fe
    collator = finetune.WhisperDataCollator(fe, tok)

    # synth 1s utterance at 16 kHz + padded reference text
    audio = np.random.randn(16000).astype(np.float32) * 0.01
    batch = collator([
        {"file": audio, "text": "سلام دنیا"},
        {"file": audio, "text": "این یک آزمون است"},
    ])

    assert batch["input_features"].shape == (2, 80, 3000)
    assert batch["labels"].shape == (2, 224)
    # padded label positions must be masked to -100
    assert (batch["labels"] == -100).any()
    assert batch["labels"].dtype == torch.long
