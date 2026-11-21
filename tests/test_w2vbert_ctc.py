import torch

from persian_asr.evaluate.metrics import bootstrap_ci
from persian_asr.train.w2vbert_ctc import build_vocab, encode, greedy_decode


def test_vocab_roundtrip():
    v = build_vocab(["سلام، دنیا", "می‌شود"])
    inv = {i: c for c, i in v.items()}
    ids = encode("سلام دنیا", v)
    assert greedy_decode([i for i in ids for _ in (0, 1)], inv) == "سلام دنیا"
    assert greedy_decode([0, ids[0], 0, ids[0]], inv) == inv[ids[0]] * 2


def test_tiny_w2vbert_forward_backward():
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertForCTC

    cfg = Wav2Vec2BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                             feature_projection_input_dim=160, vocab_size=20, pad_token_id=0, add_adapter=True,
                             num_adapter_layers=1, ctc_zero_infinity=True, conv_depthwise_kernel_size=5,
                             output_hidden_size=32)
    m = Wav2Vec2BertForCTC(cfg)
    out = m(input_features=torch.randn(2, 100, 160), attention_mask=torch.ones(2, 100, dtype=torch.long),
            labels=torch.tensor([[3, 4, 5], [6, 7, -100]]))
    out.loss.backward()
    assert torch.isfinite(out.loss)


def test_bootstrap_ci_brackets_mean():
    ci = bootstrap_ci(["سلام دنیا", "من رفتم"] * 20, ["سلام دنیا", "من رفتم خانه"] * 20, n_boot=100)
    assert ci["wer"]["lo"] <= ci["wer"]["mean"] <= ci["wer"]["hi"]


def test_lm_labels_align_with_vocab():
    from persian_asr.decode.lm_fusion import build_labels
    v = build_vocab(["سلام دنیا"])
    labels = build_labels(v)
    assert len(labels) == len(v) and labels[0] == "" and labels[v["|"]] == " " and labels[v["س"]] == "س"
