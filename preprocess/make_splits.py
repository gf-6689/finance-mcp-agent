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


def _deduplicate_stage(
    records: list[dict[str, object]],
    *,
    fields: tuple[str, ...],
    skip_blank_field: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Run one deterministic deduplication layer on ordered records.

    Records must already be sorted by date and original row id and
    must be the survivors of all previous layers. Groups are keyed by
    stock symbol plus the values of the given fields, so different
    stock symbols can never share a group. When skip_blank_field is
    given, records whose value for that field is an empty string do
    not form groups and are always kept, which lets blank URLs bypass
    the URL layer while still entering later layers. Within a group of
    two or more records, a single distinct label keeps the first
    (earliest) record and drops the rest, while conflicting labels
    drop every record in the group. Input record objects are never
    mutated and the kept records preserve the input order.

    Returns (kept_records, stage_stats) where stage_stats reports
    consistent_groups, conflicting_groups, and rows_removed for this
    layer only.
    """
    groups: dict[
        tuple[object, ...],
        list[dict[str, object]],
    ] = {}

    for record in records:
        if (
            skip_blank_field is not None
            and not record[skip_blank_field]
        ):
            continue

        group_key = (
            record["stock_symbol"],
            *(record[field] for field in fields),
        )
        groups.setdefault(group_key, []).append(record)

    consistent_keys: set[tuple[object, ...]] = set()
    conflicting_keys: set[tuple[object, ...]] = set()
    consistent_groups = 0
    conflicting_groups = 0

    for group_key, group in groups.items():
        if len(group) < 2:
            continue

        labels = {member["label"] for member in group}
        if len(labels) == 1:
            consistent_groups += 1
            consistent_keys.add(group_key)
        else:
            conflicting_groups += 1
            conflicting_keys.add(group_key)

    kept_records: list[dict[str, object]] = []
    rows_removed = 0

    for record in records:
        if (
            skip_blank_field is not None
            and not record[skip_blank_field]
        ):
            kept_records.append(record)
            continue

        group_key = (
            record["stock_symbol"],
            *(record[field] for field in fields),
        )
        group = groups[group_key]

        if group_key in conflicting_keys:
            rows_removed += 1
            continue

        if (
            group_key in consistent_keys
            and record is not group[0]
        ):
            rows_removed += 1
            continue

        kept_records.append(record)

    stage_stats = {
        "consistent_groups": consistent_groups,
        "conflicting_groups": conflicting_groups,
        "rows_removed": rows_removed,
    }
    return kept_records, stage_stats


def deduplicate_records(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Deduplicate cleaned records deterministically in three stages.

    Records are first ordered by date ascending and then by original
    row id ascending. Three key layers run strictly in order: URL,
    then title plus summary, then title alone. Every group is scoped
    by stock symbol, so different stock symbols can never share a
    group. Within a group of two or more records, a single distinct
    label keeps the earliest record and drops the rest, while
    conflicting labels drop every record in the group. Records with
    an empty URL bypass the URL layer but still enter the later
    layers. The returned records keep the date and original row order.

    The function returns retained records and deduplication
    statistics.
    """
    ordered_records = sorted(
        records,
        key=lambda record: (
            record["date"],
            record["_original_row_id"],
        ),
    )

    stages = (
        ("url", ("url",), "url"),
        ("title_summary", ("title", "summary"), None),
        ("title", ("title",), None),
    )

    remaining = ordered_records
    stage_stats = {}

    for stage_name, fields, skip_blank_field in stages:
        remaining, stage_stats[stage_name] = _deduplicate_stage(
            remaining,
            fields=fields,
            skip_blank_field=skip_blank_field,
        )

    stats: dict[str, object] = {
        "input_rows": len(records),
        "stages": stage_stats,
        "output_rows": len(remaining),
        "rows_removed_total": len(records) - len(remaining),
    }
    return remaining, stats


def _pick_cutoff_date(
    cumulative_counts: list[tuple[date, int]],
    target: float,
) -> date:
    """Choose the date whose cumulative count is closest to target.

    Each entry pairs an actual date with the number of records on or
    before it. Ties in distance are broken by choosing the earlier
    date.
    """
    return min(
        cumulative_counts,
        key=lambda pair: (abs(pair[1] - target), pair[0]),
    )[0]


def split_records_by_time(
    records: list[dict[str, object]],
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, object],
]:
    """Split deduplicated records chronologically into train, val, test.

    Records are first ordered by UTC calendar date ascending and then
    by original row id ascending. Cutoffs are chosen at whole dates so
    records from the same calendar date can never be split apart: the
    train cutoff is the date whose cumulative row count is closest to
    70% of the total and the val cutoff is the date closest to 85%,
    with ties broken by choosing the earlier date. Records on or
    before the train cutoff form train, records after it up to the val
    cutoff form val, and all remaining records form test. An empty
    record set raises ValueError.

    Returns (splits, stats) where splits maps "train", "val", and
    "test" to record lists ordered by date and original row id, and
    stats reports total_rows, the two cutoff dates, and per-split row
    counts and date ranges.
    """
    if not records:
        raise ValueError("Cannot split an empty record set")

    ordered_records = sorted(
        records,
        key=lambda record: (
            record["date"],
            record["_original_row_id"],
        ),
    )
    total_rows = len(ordered_records)

    cumulative_counts: list[tuple[date, int]] = []
    cumulative = 0

    for record in ordered_records:
        cumulative += 1
        record_date = record["date"]

        if (
            cumulative_counts
            and cumulative_counts[-1][0] == record_date
        ):
            cumulative_counts[-1] = (record_date, cumulative)
        else:
            cumulative_counts.append((record_date, cumulative))

    train_cutoff = _pick_cutoff_date(
        cumulative_counts,
        0.70 * total_rows,
    )
    val_cutoff = _pick_cutoff_date(
        cumulative_counts,
        0.85 * total_rows,
    )

    splits: dict[str, list[dict[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for record in ordered_records:
        if record["date"] <= train_cutoff:
            splits["train"].append(record)
        elif record["date"] <= val_cutoff:
            splits["val"].append(record)
        else:
            splits["test"].append(record)

    split_stats: dict[str, dict[str, object]] = {}

    for name, split_records in splits.items():
        if split_records:
            start_date: object = split_records[0]["date"].isoformat()
            end_date: object = split_records[-1]["date"].isoformat()
        else:
            start_date = None
            end_date = None

        split_stats[name] = {
            "rows": len(split_records),
            "start_date": start_date,
            "end_date": end_date,
        }

    stats: dict[str, object] = {
        "total_rows": total_rows,
        "cutoffs": {
            "train_end": train_cutoff.isoformat(),
            "val_end": val_cutoff.isoformat(),
        },
        "splits": split_stats,
    }
    return splits, stats
