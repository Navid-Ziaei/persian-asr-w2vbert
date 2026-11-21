"""KenLM n-gram shallow fusion for the w2v-BERT CTC model (via pyctcdecode).

pyctcdecode / kenlm are imported lazily so the rest of the package works without them.
"""
from typing import Optional

from persian_asr.train.w2vbert_ctc import SPACE

UNK_LABEL = "⁇"  # never matched by the LM


def build_labels(vocab: dict) -> list:
    """Map our vocab ({'[PAD]':0,'[UNK]':1,'|':k,...}) to pyctcdecode labels, index-aligned.

    blank -> "", space -> " ", unk -> UNK_LABEL, chars unchanged.
    """
    labels = [""] * len(vocab)
    for tok, i in vocab.items():
        if tok == "[PAD]":
            labels[i] = ""
        elif tok == "[UNK]":
            labels[i] = UNK_LABEL
        elif tok == SPACE:
            labels[i] = " "
        else:
            labels[i] = tok
    return labels


def make_decoder(vocab: dict, kenlm_path: Optional[str] = None, alpha: float = 0.5, beta: float = 1.0):
    """alpha = LM weight, beta = word-insertion bonus; tune both on DEV, never on test."""
    from pyctcdecode import build_ctcdecoder

    return build_ctcdecoder(build_labels(vocab), kenlm_model_path=kenlm_path, alpha=alpha, beta=beta)


def decode_batch(decoder, log_probs, beam_width: int = 64) -> list:
    """log_probs: iterable of [T, V] float arrays (log-softmax, trimmed to true length)."""
    return [" ".join(decoder.decode(lp, beam_width=beam_width).split()) for lp in log_probs]
