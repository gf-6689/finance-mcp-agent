import importlib
from datetime import date


def test_normalize_text_uses_nfkc_casefold_and_collapses_whitespace():
    module = importlib.import_module("preprocess.make_splits")

    assert hasattr(module, "normalize_text"), (
        "normalize_text must be implemented in preprocess.make_splits"
    )

    assert module.normalize_text("  ＡＰＰＬＥ\tShares\nRISE  ") == "apple shares rise"


def test_normalize_stock_symbol_uses_nfkc_strip_and_upper():
    module = importlib.import_module("preprocess.make_splits")

    assert hasattr(module, "normalize_stock_symbol"), (
        "normalize_stock_symbol must be implemented in preprocess.make_splits"
    )

    assert module.normalize_stock_symbol("  ａａｐｌ \n") == "AAPL"


def test_normalize_url_uses_nfkc_and_strip_without_casefolding():
    module = importlib.import_module("preprocess.make_splits")

    assert hasattr(module, "normalize_url"), (
        "normalize_url must be implemented in preprocess.make_splits"
    )

    raw_url = "  https://Example.com/Ｐａｔｈ?Key=Value  "
    expected = "https://Example.com/Path?Key=Value"

    assert module.normalize_url(raw_url) == expected


def test_parse_label_accepts_integer_like_values():
    module = importlib.import_module("preprocess.make_splits")

    assert hasattr(module, "parse_label"), (
        "parse_label must be implemented in preprocess.make_splits"
    )

    assert module.parse_label("4.0") == 4
    assert module.parse_label(2) == 2


def test_parse_label_rejects_fractional_values():
    module = importlib.import_module("preprocess.make_splits")

    assert module.parse_label("2.5") is None


def test_parse_label_rejects_values_outside_one_to_five():
    module = importlib.import_module("preprocess.make_splits")

    assert module.parse_label("0.0") is None
    assert module.parse_label(6) is None


def test_parse_label_rejects_missing_nonfinite_and_malformed_values():
    module = importlib.import_module("preprocess.make_splits")

    invalid_values = [None, "", "nan", "not-a-label"]

    for value in invalid_values:
        assert module.parse_label(value) is None


def test_make_sample_id_uses_normalized_symbol_title_and_summary():
    module = importlib.import_module("preprocess.make_splits")

    assert hasattr(module, "make_sample_id"), (
        "make_sample_id must be implemented in preprocess.make_splits"
    )

    sample_id = module.make_sample_id(
        stock_symbol="  ａａｐｌ ",
        title=" Apple\tRISES ",
        summary="  Strong\nEARNINGS ",
    )

    assert sample_id == (
        "29f5a7bbd5f091d1dadea23476a17b0d"
        "029fb13ba1095bfe77394f6faa7cafaf"
    )


def test_parse_utc_date_converts_timestamp_to_utc_calendar_date():
    module = importlib.import_module("preprocess.make_splits")

    assert hasattr(module, "parse_utc_date"), (
        "parse_utc_date must be implemented in preprocess.make_splits"
    )

    assert module.parse_utc_date(
        "2023-12-31T23:30:00-02:00"
    ) == date(2024, 1, 1)

    assert module.parse_utc_date(
        "2023-12-16 04:00:00 UTC"
    ) == date(2023, 12, 16)


def test_parse_utc_date_rejects_missing_invalid_and_naive_values():
    module = importlib.import_module("preprocess.make_splits")

    invalid_values = [
        None,
        "",
        "not-a-date",
        "2023-12-16 04:00:00",
    ]

    for value in invalid_values:
        assert module.parse_utc_date(value) is None
