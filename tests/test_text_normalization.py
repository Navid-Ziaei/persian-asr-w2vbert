from persian_asr.data.text_normalization import normalize_text


def test_arabic_to_persian():
    assert normalize_text("يك كتاب") == "یک کتاب"


def test_nimfasele_replacement():
    assert normalize_text("میروم \u200cخانه", replace_nimfasele=True) == "میروم خانه"


def test_multi_whitespace():
    assert normalize_text("  سلام   دنیا  ") == "سلام دنیا"

def test_scoring_strips_punctuation_and_zwnj():
    from persian_asr.data.text_normalization import normalize_for_scoring
    assert normalize_for_scoring("سلام، دنیا؟") == "سلام دنیا"
    assert normalize_for_scoring("می‌شود.") == "می شود"
    assert normalize_for_scoring("«۱۲۳» كتاب") == "123 کتاب"


def test_metrics_formatting_invariant():
    from persian_asr.evaluate.metrics import compute_metrics
    m = compute_metrics(["سلام، دنیا؟"], ["سلام دنیا"])
    assert m["cer"] == 0 and m["wer"] == 0
