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


def test_clean_raw_row_standardizes_a_valid_record():
    module = importlib.import_module("preprocess.make_splits")

    assert hasattr(module, "clean_raw_row"), (
        "clean_raw_row must be implemented in preprocess.make_splits"
    )

    raw_row = {
        "Date": "2023-12-16 04:00:00 UTC",
        "Article_title": " Apple\tRISES ",
        "Lsa_summary": "  Strong\nEARNINGS ",
        "Stock_symbol": " ａａｐｌ ",
        "Url": " https://Example.com/Ｐａｔｈ ",
        "risk_deepseek": "4.0",
        "Article": "This field must not enter the split.",
    }

    record, reasons = module.clean_raw_row(
        raw_row,
        label_column="risk_deepseek",
        original_row_id=7,
    )

    assert reasons == ()
    assert record == {
        "sample_id": (
            "29f5a7bbd5f091d1dadea23476a17b0d"
            "029fb13ba1095bfe77394f6faa7cafaf"
        ),
        "date": date(2023, 12, 16),
        "title": "apple rises",
        "summary": "strong earnings",
        "stock_symbol": "AAPL",
        "url": "https://Example.com/Path",
        "label": 4,
        "_original_row_id": 7,
    }


def test_clean_raw_row_reports_all_missing_required_fields():
    module = importlib.import_module("preprocess.make_splits")

    record, reasons = module.clean_raw_row(
        {},
        label_column="risk_deepseek",
        original_row_id=8,
    )

    assert record is None
    assert reasons == (
        "missing_title",
        "missing_summary",
        "missing_stock_symbol",
        "missing_date",
        "missing_label",
    )


def test_clean_raw_row_reports_invalid_date_and_label():
    module = importlib.import_module("preprocess.make_splits")

    raw_row = {
        "Date": "not-a-date",
        "Article_title": "Apple rises",
        "Lsa_summary": "Strong earnings",
        "Stock_symbol": "AAPL",
        "Url": "",
        "risk_deepseek": "2.5",
    }

    record, reasons = module.clean_raw_row(
        raw_row,
        label_column="risk_deepseek",
        original_row_id=9,
    )

    assert record is None
    assert reasons == (
        "invalid_date",
        "invalid_label",
    )