import csv
import importlib
from datetime import date

import pytest


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


def test_load_and_clean_csv_aggregates_records_and_reason_counts(tmp_path):
    module = importlib.import_module("preprocess.make_splits")
    source_path = tmp_path / "risk.csv"

    fieldnames = [
        "Date",
        "Article_title",
        "Lsa_summary",
        "Stock_symbol",
        "Url",
        "risk_deepseek",
    ]
    rows = [
        {
            "Date": "2023-01-01 00:00:00 UTC",
            "Article_title": "Valid title",
            "Lsa_summary": "Valid summary",
            "Stock_symbol": "AAPL",
            "Url": "https://example.com/valid",
            "risk_deepseek": "4.0",
        },
        {
            "Date": "2023-01-02 00:00:00 UTC",
            "Article_title": "Missing summary",
            "Lsa_summary": "",
            "Stock_symbol": "MSFT",
            "Url": "https://example.com/missing",
            "risk_deepseek": "3.0",
        },
        {
            "Date": "2023-01-03 00:00:00 UTC",
            "Article_title": "Invalid label",
            "Lsa_summary": "Summary",
            "Stock_symbol": "NVDA",
            "Url": "https://example.com/invalid",
            "risk_deepseek": "0.0",
        },
    ]

    with source_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    assert hasattr(module, "load_and_clean_csv"), (
        "load_and_clean_csv must be implemented"
    )

    records, stats = module.load_and_clean_csv(
        source_path,
        label_column="risk_deepseek",
    )

    assert len(records) == 1
    assert records[0]["title"] == "valid title"
    assert records[0]["_original_row_id"] == 0

    assert stats == {
        "raw_rows": 3,
        "rows_after_cleaning": 1,
        "dropped_rows": 2,
        "reason_counts": {
            "missing_summary": 1,
            "invalid_label": 1,
        },
    }


def test_load_and_clean_csv_rejects_missing_required_columns(tmp_path):
    module = importlib.import_module("preprocess.make_splits")
    source_path = tmp_path / "missing-columns.csv"
    source_path.write_text(
        "Date,Article_title\n"
        "2023-01-01 00:00:00 UTC,Title\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Missing required CSV columns: "
            "Lsa_summary, Stock_symbol, risk_deepseek"
        ),
    ):
        module.load_and_clean_csv(
            source_path,
            label_column="risk_deepseek",
        )


def test_deduplicate_records_keeps_earliest_consistent_url_duplicate():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "later",
            "date": date(2023, 1, 2),
            "title": "later title",
            "summary": "later summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/article",
            "label": 4,
            "_original_row_id": 20,
        },
        {
            "sample_id": "earlier",
            "date": date(2023, 1, 1),
            "title": "earlier title",
            "summary": "earlier summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/article",
            "label": 4,
            "_original_row_id": 30,
        },
    ]

    assert hasattr(module, "deduplicate_records"), (
        "deduplicate_records must be implemented"
    )

    deduplicated, _stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["earlier"]


def test_deduplicate_records_removes_entire_conflicting_url_group():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "conflict-later",
            "date": date(2023, 1, 2),
            "title": "later conflict title",
            "summary": "later conflict summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/conflict",
            "label": 4,
            "_original_row_id": 20,
        },
        {
            "sample_id": "unrelated",
            "date": date(2023, 1, 3),
            "title": "unrelated title",
            "summary": "unrelated summary",
            "stock_symbol": "MSFT",
            "url": "https://example.com/unrelated",
            "label": 3,
            "_original_row_id": 30,
        },
        {
            "sample_id": "conflict-earlier",
            "date": date(2023, 1, 1),
            "title": "earlier conflict title",
            "summary": "earlier conflict summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/conflict",
            "label": 2,
            "_original_row_id": 10,
        },
    ]

    deduplicated, _stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["unrelated"]


def test_deduplicate_records_scopes_title_summary_by_stock():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "aapl-later",
            "date": date(2023, 1, 2),
            "title": "shared title",
            "summary": "shared summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/aapl-later",
            "label": 3,
            "_original_row_id": 20,
        },
        {
            "sample_id": "msft",
            "date": date(2023, 1, 3),
            "title": "shared title",
            "summary": "shared summary",
            "stock_symbol": "MSFT",
            "url": "https://example.com/msft",
            "label": 3,
            "_original_row_id": 30,
        },
        {
            "sample_id": "aapl-earlier",
            "date": date(2023, 1, 1),
            "title": "shared title",
            "summary": "shared summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/aapl-earlier",
            "label": 3,
            "_original_row_id": 10,
        },
    ]

    deduplicated, _stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == [
        "aapl-earlier",
        "msft",
    ]


def test_deduplicate_records_removes_conflicting_title_summary_group():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "conflict-later",
            "date": date(2023, 1, 2),
            "title": "conflicting title",
            "summary": "conflicting summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/conflict-later",
            "label": 4,
            "_original_row_id": 20,
        },
        {
            "sample_id": "unrelated",
            "date": date(2023, 1, 3),
            "title": "unrelated title",
            "summary": "unrelated summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/unrelated",
            "label": 3,
            "_original_row_id": 30,
        },
        {
            "sample_id": "conflict-earlier",
            "date": date(2023, 1, 1),
            "title": "conflicting title",
            "summary": "conflicting summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/conflict-earlier",
            "label": 2,
            "_original_row_id": 10,
        },
    ]

    deduplicated, _stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["unrelated"]


def test_deduplicate_records_scopes_title_by_stock():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "aapl-later",
            "date": date(2023, 1, 2),
            "title": "shared title",
            "summary": "later aapl summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/aapl-later",
            "label": 3,
            "_original_row_id": 20,
        },
        {
            "sample_id": "msft",
            "date": date(2023, 1, 3),
            "title": "shared title",
            "summary": "msft summary",
            "stock_symbol": "MSFT",
            "url": "https://example.com/msft",
            "label": 3,
            "_original_row_id": 30,
        },
        {
            "sample_id": "aapl-earlier",
            "date": date(2023, 1, 1),
            "title": "shared title",
            "summary": "earlier aapl summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/aapl-earlier",
            "label": 3,
            "_original_row_id": 10,
        },
    ]

    deduplicated, _stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == [
        "aapl-earlier",
        "msft",
    ]


def test_deduplicate_records_removes_conflicting_title_group():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "conflict-later",
            "date": date(2023, 1, 2),
            "title": "conflicting title",
            "summary": "later conflict summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/conflict-later",
            "label": 4,
            "_original_row_id": 20,
        },
        {
            "sample_id": "unrelated",
            "date": date(2023, 1, 3),
            "title": "unrelated title",
            "summary": "unrelated summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/unrelated",
            "label": 3,
            "_original_row_id": 30,
        },
        {
            "sample_id": "conflict-earlier",
            "date": date(2023, 1, 1),
            "title": "conflicting title",
            "summary": "earlier conflict summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/conflict-earlier",
            "label": 2,
            "_original_row_id": 10,
        },
    ]

    deduplicated, _stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["unrelated"]


def test_deduplicate_records_keeps_blank_url_records_in_title_layer():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "blank-url-later",
            "date": date(2023, 1, 2),
            "title": "shared title",
            "summary": "later summary",
            "stock_symbol": "AAPL",
            "url": "",
            "label": 3,
            "_original_row_id": 20,
        },
        {
            "sample_id": "blank-url-earlier",
            "date": date(2023, 1, 1),
            "title": "shared title",
            "summary": "earlier summary",
            "stock_symbol": "AAPL",
            "url": "",
            "label": 3,
            "_original_row_id": 10,
        },
    ]

    deduplicated, _stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["blank-url-earlier"]


def test_deduplicate_records_applies_url_before_title():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "a",
            "date": date(2023, 1, 1),
            "title": "shared title",
            "summary": "summary a",
            "stock_symbol": "AAPL",
            "url": "https://example.com/shared-url",
            "label": 1,
            "_original_row_id": 10,
        },
        {
            "sample_id": "b",
            "date": date(2023, 1, 2),
            "title": "other title",
            "summary": "summary b",
            "stock_symbol": "AAPL",
            "url": "https://example.com/shared-url",
            "label": 2,
            "_original_row_id": 20,
        },
        {
            "sample_id": "c",
            "date": date(2023, 1, 3),
            "title": "shared title",
            "summary": "summary c",
            "stock_symbol": "AAPL",
            "url": "https://example.com/c-only",
            "label": 3,
            "_original_row_id": 30,
        },
    ]

    deduplicated, _stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["c"]


def test_deduplicate_records_applies_title_summary_before_title():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "a",
            "date": date(2023, 1, 1),
            "title": "shared title",
            "summary": "shared summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/a",
            "label": 1,
            "_original_row_id": 10,
        },
        {
            "sample_id": "b",
            "date": date(2023, 1, 2),
            "title": "shared title",
            "summary": "shared summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/b",
            "label": 2,
            "_original_row_id": 20,
        },
        {
            "sample_id": "c",
            "date": date(2023, 1, 3),
            "title": "shared title",
            "summary": "different summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/c",
            "label": 3,
            "_original_row_id": 30,
        },
    ]

    deduplicated, _stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["c"]


def test_deduplicate_records_reports_per_stage_statistics():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "a1",
            "date": date(2023, 1, 1),
            "title": "url consistent title one",
            "summary": "summary a1",
            "stock_symbol": "AAPL",
            "url": "https://example.com/url-consistent",
            "label": 3,
            "_original_row_id": 10,
        },
        {
            "sample_id": "a2",
            "date": date(2023, 1, 2),
            "title": "url consistent title two",
            "summary": "summary a2",
            "stock_symbol": "AAPL",
            "url": "https://example.com/url-consistent",
            "label": 3,
            "_original_row_id": 20,
        },
        {
            "sample_id": "b1",
            "date": date(2023, 1, 3),
            "title": "url conflict title one",
            "summary": "summary b1",
            "stock_symbol": "AAPL",
            "url": "https://example.com/url-conflict",
            "label": 1,
            "_original_row_id": 30,
        },
        {
            "sample_id": "b2",
            "date": date(2023, 1, 4),
            "title": "url conflict title two",
            "summary": "summary b2",
            "stock_symbol": "AAPL",
            "url": "https://example.com/url-conflict",
            "label": 2,
            "_original_row_id": 40,
        },
        {
            "sample_id": "c1",
            "date": date(2023, 1, 5),
            "title": "ts consistent title",
            "summary": "ts consistent summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/ts-consistent-1",
            "label": 3,
            "_original_row_id": 50,
        },
        {
            "sample_id": "c2",
            "date": date(2023, 1, 6),
            "title": "ts consistent title",
            "summary": "ts consistent summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/ts-consistent-2",
            "label": 3,
            "_original_row_id": 60,
        },
        {
            "sample_id": "d1",
            "date": date(2023, 1, 7),
            "title": "ts conflict title",
            "summary": "ts conflict summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/ts-conflict-1",
            "label": 4,
            "_original_row_id": 70,
        },
        {
            "sample_id": "d2",
            "date": date(2023, 1, 8),
            "title": "ts conflict title",
            "summary": "ts conflict summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/ts-conflict-2",
            "label": 5,
            "_original_row_id": 80,
        },
        {
            "sample_id": "e1",
            "date": date(2023, 1, 9),
            "title": "title consistent title",
            "summary": "summary e1",
            "stock_symbol": "AAPL",
            "url": "https://example.com/title-consistent-1",
            "label": 2,
            "_original_row_id": 90,
        },
        {
            "sample_id": "e2",
            "date": date(2023, 1, 10),
            "title": "title consistent title",
            "summary": "summary e2",
            "stock_symbol": "AAPL",
            "url": "https://example.com/title-consistent-2",
            "label": 2,
            "_original_row_id": 100,
        },
        {
            "sample_id": "f1",
            "date": date(2023, 1, 11),
            "title": "title conflict title",
            "summary": "summary f1",
            "stock_symbol": "AAPL",
            "url": "https://example.com/title-conflict-1",
            "label": 1,
            "_original_row_id": 110,
        },
        {
            "sample_id": "f2",
            "date": date(2023, 1, 12),
            "title": "title conflict title",
            "summary": "summary f2",
            "stock_symbol": "AAPL",
            "url": "https://example.com/title-conflict-2",
            "label": 5,
            "_original_row_id": 120,
        },
        {
            "sample_id": "unique",
            "date": date(2023, 1, 13),
            "title": "unique title",
            "summary": "unique summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/unique",
            "label": 4,
            "_original_row_id": 130,
        },
    ]

    deduplicated, stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["a1", "c1", "e1", "unique"]

    assert stats == {
        "input_rows": 13,
        "stages": {
            "url": {
                "consistent_groups": 1,
                "conflicting_groups": 1,
                "rows_removed": 3,
            },
            "title_summary": {
                "consistent_groups": 1,
                "conflicting_groups": 1,
                "rows_removed": 3,
            },
            "title": {
                "consistent_groups": 1,
                "conflicting_groups": 1,
                "rows_removed": 3,
            },
        },
        "output_rows": 4,
        "rows_removed_total": 9,
    }

    assert sum(
        stage["rows_removed"]
        for stage in stats["stages"].values()
    ) == stats["rows_removed_total"]

    assert (
        stats["input_rows"] - stats["output_rows"]
        == stats["rows_removed_total"]
    )


def _make_record(sample_id, record_date, row_id):
    """Build one minimal deduplicated record for split tests."""
    return {
        "sample_id": sample_id,
        "date": record_date,
        "title": f"title {sample_id}",
        "summary": f"summary {sample_id}",
        "stock_symbol": "AAPL",
        "url": f"https://example.com/{sample_id}",
        "label": 3,
        "_original_row_id": row_id,
    }


def _make_scrambled_split_records():
    """Build 20 records over five dates in scrambled input order.

    Cumulative counts by date are 13, 15, 16, 18, 20, so the 70%
    target of 14 ties between the first two dates and the 85% target
    of 17 ties between the third and fourth dates; the earlier date
    must win both ties.
    """
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    date3 = date(2023, 1, 3)
    date4 = date(2023, 1, 4)
    date5 = date(2023, 1, 5)

    scrambled = [
        ("d5-02", date5),
        ("d1-03", date1),
        ("d4-01", date4),
        ("d2-02", date2),
        ("d1-01", date1),
        ("d3-01", date3),
        ("d5-01", date5),
        ("d1-07", date1),
        ("d4-02", date4),
        ("d1-02", date1),
        ("d2-01", date2),
        ("d1-13", date1),
        ("d1-05", date1),
        ("d1-12", date1),
        ("d1-04", date1),
        ("d1-09", date1),
        ("d1-06", date1),
        ("d1-11", date1),
        ("d1-10", date1),
        ("d1-08", date1),
    ]

    return [
        _make_record(sample_id, record_date, row_id)
        for row_id, (sample_id, record_date) in enumerate(scrambled)
    ]


def _sorted_sample_ids(records):
    """Return sample ids of records ordered by date and row id."""
    return [
        record["sample_id"]
        for record in sorted(
            records,
            key=lambda record: (
                record["date"],
                record["_original_row_id"],
            ),
        )
    ]


def test_split_records_by_time_keeps_dates_intact():
    module = importlib.import_module("preprocess.make_splits")

    assert hasattr(module, "split_records_by_time"), (
        "split_records_by_time must be implemented"
    )

    records = _make_scrambled_split_records()

    splits, _stats = module.split_records_by_time(records)

    assert len(splits["train"]) == 13
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 4

    assert {
        record["date"] for record in splits["train"]
    } == {date(2023, 1, 1)}

    assert {
        record["date"] for record in splits["val"]
    } == {date(2023, 1, 2), date(2023, 1, 3)}

    assert {
        record["date"] for record in splits["test"]
    } == {date(2023, 1, 4), date(2023, 1, 5)}

    train_dates = {
        record["date"] for record in splits["train"]
    }
    val_dates = {
        record["date"] for record in splits["val"]
    }
    test_dates = {
        record["date"] for record in splits["test"]
    }

    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)

    assert (
        len(splits["train"])
        + len(splits["val"])
        + len(splits["test"])
        == 20
    )


def test_split_records_by_time_uses_earlier_date_when_cutoff_distance_ties():
    module = importlib.import_module("preprocess.make_splits")

    records = _make_scrambled_split_records()

    _splits, stats = module.split_records_by_time(records)

    assert stats["cutoffs"]["train_end"] == "2023-01-01"
    assert stats["cutoffs"]["val_end"] == "2023-01-03"


def test_split_records_by_time_sorts_by_date_and_original_row_id():
    module = importlib.import_module("preprocess.make_splits")

    records = _make_scrambled_split_records()

    splits, _stats = module.split_records_by_time(records)

    expected_train = _sorted_sample_ids(
        record
        for record in records
        if record["date"] == date(2023, 1, 1)
    )
    expected_val = _sorted_sample_ids(
        record
        for record in records
        if record["date"] in (date(2023, 1, 2), date(2023, 1, 3))
    )
    expected_test = _sorted_sample_ids(
        record
        for record in records
        if record["date"] in (date(2023, 1, 4), date(2023, 1, 5))
    )

    assert [
        record["sample_id"]
        for record in splits["train"]
    ] == expected_train

    assert [
        record["sample_id"]
        for record in splits["val"]
    ] == expected_val

    assert [
        record["sample_id"]
        for record in splits["test"]
    ] == expected_test

    assert (
        _sorted_sample_ids(splits["train"]) == expected_train
    )


def test_split_records_by_time_rejects_empty_input():
    module = importlib.import_module("preprocess.make_splits")

    with pytest.raises(
        ValueError,
        match="Cannot split an empty record set",
    ):
        module.split_records_by_time([])


def test_split_records_by_time_reports_cutoffs_counts_and_ranges():
    module = importlib.import_module("preprocess.make_splits")

    records = _make_scrambled_split_records()

    splits, stats = module.split_records_by_time(records)

    assert stats == {
        "total_rows": 20,
        "cutoffs": {
            "train_end": "2023-01-01",
            "val_end": "2023-01-03",
        },
        "splits": {
            "train": {
                "rows": 13,
                "start_date": "2023-01-01",
                "end_date": "2023-01-01",
            },
            "val": {
                "rows": 3,
                "start_date": "2023-01-02",
                "end_date": "2023-01-03",
            },
            "test": {
                "rows": 4,
                "start_date": "2023-01-04",
                "end_date": "2023-01-05",
            },
        },
    }

    for name, split_records in splits.items():
        assert stats["splits"][name]["rows"] == len(split_records)

    assert (
        sum(
            split_info["rows"]
            for split_info in stats["splits"].values()
        )
        == stats["total_rows"]
    )


def test_split_records_by_time_reports_none_dates_for_empty_splits():
    module = importlib.import_module("preprocess.make_splits")

    single_date = date(2023, 1, 1)
    records = [
        _make_record(f"single-{index:02d}", single_date, index)
        for index in range(10)
    ]

    splits, stats = module.split_records_by_time(records)

    assert len(splits["train"]) == 10
    assert splits["val"] == []
    assert splits["test"] == []

    assert stats["splits"]["val"] == {
        "rows": 0,
        "start_date": None,
        "end_date": None,
    }
    assert stats["splits"]["test"] == {
        "rows": 0,
        "start_date": None,
        "end_date": None,
    }


def test_deduplicate_records_does_not_group_blank_urls():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "blank-a",
            "date": date(2023, 1, 1),
            "title": "blank url title a",
            "summary": "blank url summary a",
            "stock_symbol": "AAPL",
            "url": "",
            "label": 1,
            "_original_row_id": 10,
        },
        {
            "sample_id": "blank-b",
            "date": date(2023, 1, 2),
            "title": "blank url title b",
            "summary": "blank url summary b",
            "stock_symbol": "AAPL",
            "url": "",
            "label": 2,
            "_original_row_id": 20,
        },
    ]

    deduplicated, stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["blank-a", "blank-b"]

    assert stats["stages"]["url"] == {
        "consistent_groups": 0,
        "conflicting_groups": 0,
        "rows_removed": 0,
    }


def test_deduplicate_records_scopes_url_by_stock():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "aapl-url",
            "date": date(2023, 1, 1),
            "title": "aapl title",
            "summary": "aapl summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/shared-url",
            "label": 1,
            "_original_row_id": 10,
        },
        {
            "sample_id": "msft-url",
            "date": date(2023, 1, 2),
            "title": "msft title",
            "summary": "msft summary",
            "stock_symbol": "MSFT",
            "url": "https://example.com/shared-url",
            "label": 2,
            "_original_row_id": 20,
        },
    ]

    deduplicated, stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["aapl-url", "msft-url"]

    assert stats["stages"]["url"] == {
        "consistent_groups": 0,
        "conflicting_groups": 0,
        "rows_removed": 0,
    }


def test_deduplicate_records_reports_stats_from_stage_survivors():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "a",
            "date": date(2023, 1, 1),
            "title": "ts shared title",
            "summary": "ts shared summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/a-b-shared",
            "label": 1,
            "_original_row_id": 10,
        },
        {
            "sample_id": "b",
            "date": date(2023, 1, 2),
            "title": "b unique title",
            "summary": "b unique summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/a-b-shared",
            "label": 2,
            "_original_row_id": 20,
        },
        {
            "sample_id": "c",
            "date": date(2023, 1, 3),
            "title": "ts shared title",
            "summary": "ts shared summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/c",
            "label": 3,
            "_original_row_id": 30,
        },
        {
            "sample_id": "d",
            "date": date(2023, 1, 4),
            "title": "title shared title",
            "summary": "title shared summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/d",
            "label": 4,
            "_original_row_id": 40,
        },
        {
            "sample_id": "e",
            "date": date(2023, 1, 5),
            "title": "title shared title",
            "summary": "title shared summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/e",
            "label": 5,
            "_original_row_id": 50,
        },
        {
            "sample_id": "f",
            "date": date(2023, 1, 6),
            "title": "title shared title",
            "summary": "f unique summary",
            "stock_symbol": "AAPL",
            "url": "https://example.com/f",
            "label": 3,
            "_original_row_id": 60,
        },
    ]

    deduplicated, stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["c", "f"]

    assert stats["stages"]["url"] == {
        "consistent_groups": 0,
        "conflicting_groups": 1,
        "rows_removed": 2,
    }

    assert stats["stages"]["title_summary"] == {
        "consistent_groups": 0,
        "conflicting_groups": 1,
        "rows_removed": 2,
    }

    assert stats["stages"]["title"] == {
        "consistent_groups": 0,
        "conflicting_groups": 0,
        "rows_removed": 0,
    }


def test_deduplicate_records_breaks_same_date_ties_by_original_row_id():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        {
            "sample_id": "same-date-row-20",
            "date": date(2023, 1, 1),
            "title": "tie title later",
            "summary": "tie summary later",
            "stock_symbol": "AAPL",
            "url": "https://example.com/tie-url",
            "label": 3,
            "_original_row_id": 20,
        },
        {
            "sample_id": "same-date-row-10",
            "date": date(2023, 1, 1),
            "title": "tie title earlier",
            "summary": "tie summary earlier",
            "stock_symbol": "AAPL",
            "url": "https://example.com/tie-url",
            "label": 3,
            "_original_row_id": 10,
        },
    ]

    deduplicated, _stats = module.deduplicate_records(records)

    assert [
        record["sample_id"]
        for record in deduplicated
    ] == ["same-date-row-10"]


def test_split_records_by_time_orders_same_date_by_original_row_id():
    module = importlib.import_module("preprocess.make_splits")

    records = [
        _make_record("same-day-30", date(2023, 1, 1), 30),
        _make_record("same-day-10", date(2023, 1, 1), 10),
        _make_record("same-day-20", date(2023, 1, 1), 20),
    ]

    splits, _stats = module.split_records_by_time(records)

    containing_split = next(
        split_records
        for split_records in splits.values()
        if any(
            record["sample_id"] == "same-day-10"
            for record in split_records
        )
    )

    assert [
        record["sample_id"]
        for record in containing_split
    ] == ["same-day-10", "same-day-20", "same-day-30"]
