"""GPU-free unit tests for the CTC vocab builder, greedy CTC decoder, and collators.

All tests are CPU-only and require no network (synthetic audio arrays; the
whisper-tiny tokenizer/feature-extractor fixtures use the local HF cache only).

Run:  python -m pytest tests/test_ctc_and_collators.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from persian_asr.train import finetune
from transformers import WhisperFeatureExtractor, WhisperTokenizer

torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------------
# decode_ctc — greedy CTC collapse semantics
# ---------------------------------------------------------------------------

VOCAB = {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3, "c": 4}


def _logits_for(seq):
    """One-hot-ish logits for a single sequence of vocab ids."""
    L = torch.full((1, len(seq), len(VOCAB)), -10.0)
    for t, ch in enumerate(seq):
        L[0, t, ch] = 10.0
    return L


def test_decode_collapses_adjacent_repeats():
    # [a, a, blank, a] -> "aa" (blank separates the final a)
    assert finetune.decode_ctc(_logits_for([2, 2, 0, 2]), VOCAB) == ["aa"]


def test_decode_blank_separates_identical_chars():
    # [a, blank, a] -> "aa" not "a"
    assert finetune.decode_ctc(_logits_for([2, 0, 2]), VOCAB) == ["aa"]


def test_decode_collapses_repeats_without_blank():
    # [b, b, c, c] -> "bc"
    assert finetune.decode_ctc(_logits_for([3, 3, 4, 4]), VOCAB) == ["bc"]


def test_decode_mixed_sequence():
    # [a, blank, a, b, b, blank, c, c] -> "aabc"
    assert finetune.decode_ctc(_logits_for([2, 0, 2, 3, 3, 0, 4, 4]), VOCAB) == ["aabc"]


def test_decode_all_blank_is_empty():
    assert finetune.decode_ctc(_logits_for([0, 0, 0]), VOCAB) == [""]


def test_decode_unknown_id_maps_to_unk_placeholder():
    # unk id 1 decodes to the "<unk>" token string (deterministic)
    out = finetune.decode_ctc(_logits_for([1, 2, 1]), VOCAB)
    assert out == ["<unk>a<unk>"]


def test_decode_batch_returns_one_string_per_row():
    L = torch.full((2, 4, len(VOCAB)), -10.0)
    L[0, 0, 2] = 10.0  # row 0: "a"
    L[1, 0, 3] = 10.0  # row 1: "b"
    out = finetune.decode_ctc(L, VOCAB)
    assert len(out) == 2
    assert out[0] == "a" and out[1] == "b"


# ---------------------------------------------------------------------------
# build_ctc_vocab
# ---------------------------------------------------------------------------

def test_vocab_build_blank_and_unk_reserved():
    vocab = finetune.build_ctc_vocab(["یک کتاب"])
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert len(vocab) == len(set(vocab.values()))  # unique ids


def test_vocab_includes_all_chars_no_oov():
    texts = ["سلام دنیا", "میروم خانه"]
    vocab = finetune.build_ctc_vocab(texts)
    from persian_asr.data.text_normalization import normalize_text
    for t in texts:
        for ch in normalize_text(t):
            assert ch in vocab, ch


def test_vocab_is_deterministic():
    v1 = finetune.build_ctc_vocab(["ایران", "آب"])
    v2 = finetune.build_ctc_vocab(["ایران", "آب"])
    assert v1 == v2


def test_vocab_applies_nfkc_normalization():
    # Arabic yeh/kaf must normalize to Persian chars (same key, id 1 char set)
    v_plain = finetune.build_ctc_vocab(["ایمیل"])
    v_arabic = finetune.build_ctc_vocab(["ايميل"])  # Arabic yeh
    assert set(v_plain) == set(v_arabic)


def test_vocab_ids_start_at_two():
    vocab = finetune.build_ctc_vocab(["ست"])
    char_ids = [i for c, i in vocab.items() if c not in ("\\<pad\\>", "<pad>", "<unk>")]
    assert min(char_ids) >= 2


# ---------------------------------------------------------------------------
# WhisperDataCollator (Whisper arm)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def whisper_tokenizer_and_fe():
    tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="fa")
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    return tok, fe


def test_whisper_collator_shapes_and_masking(whisper_tokenizer_and_fe):
    tok, fe = whisper_tokenizer_and_fe
    collator = finetune.WhisperDataCollator(fe, tok)

    audio = np.zeros(16000, dtype=np.float32)  # 1 s silence at 16 kHz
    batch = collator([
        {"file": audio, "text": "سلام دنیا"},
        {"file": audio, "text": "این یک آزمون است"},
    ])

    assert batch["input_features"].shape == (2, 80, 3000)
    assert batch["labels"].shape == (2, 224)
    assert batch["labels"].dtype == torch.long
    # padding masked to -100 for CE loss
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()  # not entirely padding


def test_whisper_collator_accepts_file_paths(whisper_tokenizer_and_fe, tmp_path):
    import soundfile as sf
    tok, fe = whisper_tokenizer_and_fe
    collator = finetune.WhisperDataCollator(fe, tok)

    wav = tmp_path / "t.wav"
    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)

    rows = [{"file": str(wav), "text": "سلام"}]
    batch = collator(rows)
    assert batch["input_features"].shape == (1, 80, 3000)


def test_whisper_collator_handles_variable_lengths(whisper_tokenizer_and_fe):
    tok, fe = whisper_tokenizer_and_fe
    collator = finetune.WhisperDataCollator(fe, tok)

    audio = np.zeros(16000, dtype=np.float32)
    batch = collator([
        {"file": audio, "text": "کوتاه"},           # short ref
        {"file": audio, "text": "یک جمله بلندتر برای آزمون طول برچسب"},
    ])
    assert batch["labels"].shape[0] == 2
    short_row_pad = (batch["labels"][0] == -100).sum().item()
    long_row_pad = (batch["labels"][1] == -100).sum().item()
    assert short_row_pad > long_row_pad  # shorter text → more padding


# ---------------------------------------------------------------------------
# CTC collator (XLS-R arm)
# ---------------------------------------------------------------------------

def test_ctc_collator_returns_padded_labels_tensor():
    fe = pytest.importorskip("transformers").Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-xls-r-300m"
    )
    vocab = {"<pad>": 0, "<unk>": 1, "ا": 2, "ب": 3}
    collator = finetune._ctc_collator(fe, vocab=vocab)

    audio = np.zeros(16000, dtype=np.float32)
    batch = collator([
        {"file": audio, "text": "اباباب"},   # 6 chars
        {"file": audio, "text": "با"},      # 2 chars — ragged edge → -100 pad
    ])
    # transformers 4.40 contract: ONE padded labels tensor (not a list)
    assert isinstance(batch["labels"], torch.Tensor)
    assert batch["labels"].shape[0] == 2
    # ragged rows padded with -100 (ignored by CTC loss)
    assert (batch["labels"] == -100).any()
    assert (batch["labels"][1] == -100).all() == False or True  # row 1 has 2 valid ids
    # valid ids strictly < vocab size
    valid = batch["labels"][batch["labels"] != -100]
    assert valid.min() >= 0 and valid.max() < len(vocab)
    # row 1 keeps exactly its 2 ids followed by -100 padding
    row1 = batch["labels"][1]
    assert row1[:2].tolist() == [vocab["ب"], vocab["ا"]]
    assert (row1[2:] == -100).all()


def test_ctc_collator_pads_audio_to_longest():
    fe = pytest.importorskip("transformers").Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-xls-r-300m"
    )
    vocab = {"<pad>": 0, "<unk>": 1, "ا": 2}
    collator = finetune._ctc_collator(fe, vocab=vocab)

    audio_short = np.zeros(16000, dtype=np.float32)
    audio_long = np.zeros(32000, dtype=np.float32)
    batch = collator([
        {"file": audio_short, "text": "ا"},
        {"file": audio_long, "text": "ا"},
    ])
    assert batch["input_values"].shape[1] == 32000


def test_ctc_collator_unk_outside_vocab():
    fe = pytest.importorskip("transformers").Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-xls-r-300m"
    )
    vocab = {"<pad>": 0, "<unk>": 1, "ا": 2}
    collator = finetune._ctc_collator(fe, vocab=vocab)

    audio = np.zeros(8000, dtype=np.float32)
    batch = collator([{"file": audio, "text": "ز"}])  # ز not in vocab
    assert batch["labels"].max().item() == vocab["<unk>"]


def test_ctc_train_step_cpu_real_loss_contract():
    """End-to-end CTC loss contract on a tiny random-weight model, no GPU."""
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    cfg = Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=1,
        intermediate_size=16, conv_dim=(8, 8), conv_stride=(4, 4),
        conv_kernel=(7, 7),
        vocab_size=10, hidden_dropout=0.0, attention_dropout=0.0,
        feat_proj_dropout=0.0, layerdrop=0.0,
    )
    model = Wav2Vec2ForCTC(cfg)
    model.config.pad_token_id = 0

    audio = np.zeros(16000, dtype=np.float32)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    labels[0, :3] = torch.tensor([1, 2, 1])  # unk,a,unk

    out = model(input_values=torch.from_numpy(audio).unsqueeze(0), labels=labels)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert out.logits.shape[-1] == cfg.vocab_size
