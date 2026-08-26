"""Build reproducible time-based dataset splits."""

import csv
import hashlib
import math
import unicodedata
from collections import Counter
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path


def normalize_text(value: str) -> str:
    """Normalize text for exact-match deduplication."""
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


def normalize_stock_symbol(value: str) -> str:
    """Normalize a stock symbol for task-scoped deduplication."""
    return unicodedata.normalize("NFKC", value).strip().upper()


def normalize_url(value: str) -> str:
    """Normalize a URL without changing path or query case."""
    return unicodedata.normalize("NFKC", value).strip()


def parse_label(value: object) -> int | None:
    """Convert a valid 1-5 integer-like label value to int."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed) or not parsed.is_integer():
        return None

    label = int(parsed)
    if label not in range(1, 6):
        return None

    return label


def make_sample_id(
    *,
    stock_symbol: str,
    title: str,
    summary: str,
) -> str:
    """Create a stable ID from normalized task input fields."""
    sample_key = "\n".join(
        (
            normalize_stock_symbol(stock_symbol),
            normalize_text(title),
            normalize_text(summary),
        )
    )
    return hashlib.sha256(sample_key.encode("utf-8")).hexdigest()


def parse_utc_date(value: object) -> date | None:
    """Parse a timezone-aware timestamp into its UTC calendar date."""
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    if not normalized:
        return None

    if normalized.endswith(" UTC"):
        normalized = normalized.removesuffix(" UTC") + "+00:00"

    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return None

    return timestamp.astimezone(timezone.utc).date()


def _is_missing(value: object) -> bool:
    """Return whether a source field is absent or blank."""
    return value is None or (
        isinstance(value, str) and not value.strip()
    )


def clean_raw_row(
    raw_row: Mapping[str, object],
    *,
    label_column: str,
    original_row_id: int,
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    """Convert one source row to the standard internal schema."""
    required_fields = (
        ("Article_title", "missing_title"),
        ("Lsa_summary", "missing_summary"),
        ("Stock_symbol", "missing_stock_symbol"),
        ("Date", "missing_date"),
        (label_column, "missing_label"),
    )

    reasons = tuple(
        reason
        for field, reason in required_fields
        if _is_missing(raw_row.get(field))
    )
    if reasons:
        return None, reasons

    title = normalize_text(str(raw_row["Article_title"]))
    summary = normalize_text(str(raw_row["Lsa_summary"]))
    stock_symbol = normalize_stock_symbol(
        str(raw_row["Stock_symbol"])
    )

    raw_url = raw_row.get("Url")
    url = (
        ""
        if _is_missing(raw_url)
        else normalize_url(str(raw_url))
    )

    parsed_date = parse_utc_date(raw_row["Date"])
    label = parse_label(raw_row[label_column])

    invalid_reasons = []
    if parsed_date is None:
        invalid_reasons.append("invalid_date")
    if label is None:
        invalid_reasons.append("invalid_label")

    if invalid_reasons:
        return None, tuple(invalid_reasons)

    record = {
        "sample_id": make_sample_id(
            stock_symbol=stock_symbol,
            title=title,
            summary=summary,
        ),
        "date": parsed_date,
        "title": title,
        "summary": summary,
        "stock_symbol": stock_symbol,
        "url": url,
        "label": label,
        "_original_row_id": original_row_id,
    }
    return record, ()


def load_and_clean_csv(
    source_path: str | Path,
    *,
    label_column: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load a source CSV and aggregate row-cleaning statistics."""
    records = []
    reason_counts: Counter[str] = Counter()
    raw_rows = 0

    with Path(source_path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required_columns = (
            "Date",
            "Article_title",
            "Lsa_summary",
            "Stock_symbol",
            label_column,
        )
        available_columns = set(reader.fieldnames or ())
        missing_columns = [
            column
            for column in required_columns
            if column not in available_columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required CSV columns: "
                + ", ".join(missing_columns)
            )

        for original_row_id, raw_row in enumerate(reader):
            raw_rows += 1
            record, reasons = clean_raw_row(
                raw_row,
                label_column=label_column,
                original_row_id=original_row_id,
            )

            if reasons:
                reason_counts.update(reasons)
                continue

            if record is not None:
                records.append(record)

    stats = {
        "raw_rows": raw_rows,
        "rows_after_cleaning": len(records),
        "dropped_rows": raw_rows - len(records),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    return records, stats
