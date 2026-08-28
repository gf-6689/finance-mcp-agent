"""Build reproducible time-based dataset splits."""

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import unicodedata
from collections import Counter
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path

EVAL_LABELS = (1, 2, 3, 4, 5)
CLASS_MINIMUM = 10
PROTOCOL_VERSION = "1.0.0"
SPLIT_FIELDS = (
    "sample_id",
    "date",
    "title",
    "summary",
    "stock_symbol",
    "url",
    "label",
)
TASK_LABEL_COLUMNS = {"risk": "risk_deepseek"}


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

    reason_hits = sum(reason_counts.values())
    unique_removed_rows = raw_rows - len(records)
    stats = {
        "raw_rows": raw_rows,
        "rows_after_cleaning": len(records),
        "unique_removed_rows": unique_removed_rows,
        "reason_hits": reason_hits,
        "overlap": reason_hits - unique_removed_rows,
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


def compute_eval_quotas(
    counts: Mapping[int, int],
    *,
    eval_size: int,
    labels: tuple[int, ...],
    class_minimum: int = CLASS_MINIMUM,
) -> dict[int, int]:
    """Compute deterministic per-label eval quotas (largest remainder).

    Every label that actually exists in Test first receives
    min(class_minimum, count) rows. The remaining seats are then
    shared proportionally to each label's remaining capacity, floors
    are taken first, and any leftover seats go to the labels with the
    largest fractional remainders, with ties broken by ascending
    label. A quota never exceeds the label's real count: when a label
    cannot receive any more seats the remaining seats are re-allocated
    over the labels that still have capacity, using the same largest
    remainder principle, until the total is exactly eval_size.
    """
    if eval_size < 0:
        raise ValueError("eval_size must be non-negative")

    present = sorted(
        label for label in labels if counts.get(label, 0) > 0
    )
    total_available = sum(counts[label] for label in present)

    if total_available < eval_size:
        raise ValueError(
            "Test set has fewer rows than eval_size: "
            f"{total_available} < {eval_size}"
        )

    quota = {
        label: min(class_minimum, counts[label])
        for label in present
    }
    remaining = eval_size - sum(quota.values())
    if remaining < 0:
        raise ValueError(
            "class_minimum allocation exceeds eval_size"
        )

    while remaining > 0:
        eligible = [
            label
            for label in present
            if counts[label] - quota[label] > 0
        ]
        if not eligible:
            raise ValueError(
                "Cannot reach eval_size: every label is at capacity"
            )

        total_capacity = sum(
            counts[label] - quota[label] for label in eligible
        )
        ideal = {
            label: remaining
            * (counts[label] - quota[label])
            / total_capacity
            for label in eligible
        }
        floor_extra = {
            label: math.floor(ideal[label])
            for label in eligible
        }
        for label in eligible:
            addition = min(
                floor_extra[label],
                counts[label] - quota[label],
            )
            quota[label] += addition

        remaining = eval_size - sum(quota.values())
        if remaining == 0:
            break

        order = sorted(
            eligible,
            key=lambda label: (
                -(ideal[label] - floor_extra[label]),
                label,
            ),
        )
        progressed = False
        for label in order:
            if remaining == 0:
                break
            if quota[label] < counts[label]:
                quota[label] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError(
                "Cannot reach eval_size: every label is at capacity"
            )

    return quota


def sample_eval_records(
    test_records: list[dict[str, object]],
    *,
    eval_size: int,
    labels: tuple[int, ...],
    seed: int,
) -> tuple[list[dict[str, object]], dict[int, int]]:
    """Draw a fixed deterministic stratified eval set from Test only.

    Quotas come from compute_eval_quotas. Each label's pool is ordered
    by sample id ascending before the seeded sample is drawn, so the
    result never depends on the input record order. Sampling within
    each label is without replacement and records are never copied.
    The returned records are concatenated and sorted by sample id
    ascending so the output row order is fixed.
    """
    counts: Counter[int] = Counter()
    for record in test_records:
        counts[int(record["label"])] += 1

    quotas = compute_eval_quotas(
        counts,
        eval_size=eval_size,
        labels=labels,
    )

    rng = random.Random(seed)
    sampled: list[dict[str, object]] = []

    for label in sorted(quotas):
        quota = quotas[label]
        if quota <= 0:
            continue
        pool = sorted(
            (
                record
                for record in test_records
                if record["label"] == label
            ),
            key=lambda record: record["sample_id"],
        )
        if quota >= len(pool):
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, quota))

    sampled.sort(key=lambda record: record["sample_id"])
    return sampled, quotas


def write_split_csv(
    path: str | Path,
    records: list[dict[str, object]],
) -> None:
    """Write records with the frozen field order and fixed line endings.

    The CSV is UTF-8 with "\n" line endings and the fields
    sample_id, date, title, summary, stock_symbol, url, label in that
    exact order. Fields such as _original_row_id never enter the file.
    """
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SPLIT_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            row = {field: record[field] for field in SPLIT_FIELDS}
            row["date"] = record["date"].isoformat()
            writer.writerow(row)


def sha256_file(path: str | Path) -> str:
    """Return the lowercase hex SHA256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Return the lowercase hex SHA256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def count_physical_lines(path: str | Path) -> int:
    """Count physical text lines in a file, including the header.

    This deliberately counts raw line breaks rather than parsed CSV
    rows, because quoted fields (such as the Article body) may
    contain embedded newlines.
    """
    with Path(path).open("rb") as handle:
        return sum(1 for _ in handle)


def read_csv_sample_ids(path: str | Path) -> list[str]:
    """Read sample_id values from a split CSV in file order."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [
            row["sample_id"] for row in csv.DictReader(handle)
        ]


def label_distribution(
    records: list[dict[str, object]],
) -> dict[str, int]:
    """Count records per eval label, reporting every label in 1..5."""
    counts: Counter[int] = Counter()
    for record in records:
        counts[int(record["label"])] += 1
    return {
        str(label): counts[label] for label in EVAL_LABELS
    }


def check_split_integrity(
    splits: Mapping[str, list[dict[str, object]]],
    eval_records: list[dict[str, object]],
) -> dict[str, int]:
    """Report duplicate and cross-split sample id overlap counts."""
    id_sets = {
        name: {record["sample_id"] for record in records}
        for name, records in splits.items()
    }
    eval_ids = {
        record["sample_id"] for record in eval_records
    }
    duplicate_counts = {
        name: len(records) - len(id_sets[name])
        for name, records in splits.items()
    }
    duplicate_counts["eval_test"] = (
        len(eval_records) - len(eval_ids)
    )
    return {
        "train_duplicate_sample_ids": duplicate_counts["train"],
        "val_duplicate_sample_ids": duplicate_counts["val"],
        "test_duplicate_sample_ids": duplicate_counts["test"],
        "eval_test_duplicate_sample_ids": duplicate_counts[
            "eval_test"
        ],
        "train_intersect_val": len(
            id_sets["train"] & id_sets["val"]
        ),
        "train_intersect_test": len(
            id_sets["train"] & id_sets["test"]
        ),
        "val_intersect_test": len(
            id_sets["val"] & id_sets["test"]
        ),
        "eval_test_outside_test": len(
            eval_ids - id_sets["test"]
        ),
    }


def _split_info(
    name: str,
    records: list[dict[str, object]],
    csv_path: str | Path,
) -> dict[str, object]:
    """Build one split's manifest entry including file evidence.

    Date ranges use the minimum and maximum record dates rather than
    the first and last rows, because eval_test rows are ordered by
    sample id and not by date.
    """
    sorted_ids = sorted(
        record["sample_id"] for record in records
    )
    record_dates = [record["date"] for record in records]
    return {
        "name": name,
        "rows": len(records),
        "start_date": (
            min(record_dates).isoformat() if record_dates else None
        ),
        "end_date": (
            max(record_dates).isoformat() if record_dates else None
        ),
        "label_distribution": label_distribution(records),
        "csv_sha256": sha256_file(csv_path),
        "sample_ids_sha256": sha256_text("\n".join(sorted_ids)),
    }


def git_commit_sha(repo_dir: str | Path) -> str | None:
    """Return the current Git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_working_tree_dirty(repo_dir: str | Path) -> bool | None:
    """Return whether the Git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout)


def build_manifest(
    *,
    task: str,
    source_path: str | Path,
    source_rows: int,
    source_physical_lines: int,
    eval_size: int,
    seed: int,
    output_dir: str | Path,
    splits: Mapping[str, list[dict[str, object]]],
    eval_records: list[dict[str, object]],
    cutoffs: Mapping[str, str],
    repo_dir: str | Path,
) -> dict[str, object]:
    """Build the split freeze manifest with source and file evidence."""
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    code_path = Path(__file__)
    try:
        code_path_display = code_path.resolve().relative_to(
            Path(repo_dir).resolve()
        )
    except ValueError:
        code_path_display = code_path

    split_entries = {}
    for name in ("train", "val", "test"):
        split_entries[name] = _split_info(
            name,
            splits[name],
            output_dir / f"{name}.csv",
        )
    split_entries["eval_test"] = _split_info(
        "eval_test",
        eval_records,
        output_dir / "eval_test.csv",
    )

    return {
        "task": task,
        "protocol_version": PROTOCOL_VERSION,
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "logical_csv_rows": source_rows,
            "physical_text_lines_including_header": (
                source_physical_lines
            ),
        },
        "split_rule": {
            "type": "time_boundary",
            "target_ratio": {
                "train": 0.70,
                "val": 0.15,
                "test": 0.15,
            },
            "cutoffs": dict(cutoffs),
        },
        "eval_rule": {
            "eval_size": eval_size,
            "seed": seed,
            "labels": list(EVAL_LABELS),
            "class_minimum": CLASS_MINIMUM,
            "allocation": (
                "minimum_then_remaining_capacity_largest_remainder"
            ),
            "tie_break": "label_ascending",
        },
        "splits": split_entries,
        "code": {
            "path": str(code_path_display),
            "sha256": sha256_file(code_path),
        },
        "git": {
            "commit": git_commit_sha(repo_dir),
            "working_tree_dirty": git_working_tree_dirty(repo_dir),
        },
    }


def build_report(
    *,
    task: str,
    source_path: str | Path,
    source_sha256: str,
    source_physical_lines: int,
    clean_stats: Mapping[str, object],
    dedup_stats: Mapping[str, object],
    split_stats: Mapping[str, object],
    eval_size: int,
    seed: int,
    quotas: Mapping[int, int],
    split_label_dists: Mapping[str, Mapping[str, int]],
    integrity: Mapping[str, int],
    verification: Mapping[str, Mapping[str, bool]] | None,
) -> str:
    """Render the split freeze report as Markdown."""
    lines: list[str] = [
        "# Risk Split Freeze Report",
        "",
        "## Source",
        "",
        f"- 源文件：`{source_path}`",
        f"- SHA256：`{source_sha256}`",
        f"- logical CSV rows: {clean_stats['raw_rows']}",
        "- physical text lines including header: "
        f"{source_physical_lines}",
        "- 行数口径：原始 Risk 数据包含 "
        f"{clean_stats['raw_rows']} 条金融新闻记录；"
        "物理文本行数更多，因为 `Article` 字段的引号内"
        "嵌换行被计为文本行而非 CSV 记录。",
        "",
        "## Cleaning",
        "",
        f"- raw rows: {clean_stats['raw_rows']}",
        f"- after cleaning: {clean_stats['rows_after_cleaning']}",
        f"- unique removed rows: {clean_stats['unique_removed_rows']}",
        f"- reason hits: {clean_stats['reason_hits']}",
        f"- overlap: {clean_stats['overlap']}",
        f"- after dedup: {dedup_stats['output_rows']}",
        f"- dropped reason counts: "
        f"{json.dumps(clean_stats['reason_counts'], ensure_ascii=False)}",
        "",
        "## Dedup",
        "",
        "> 当前只进行了股票任务粒度的精确去重和时间切分，"
        "不声称完成事件级近似去重。",
        "",
    ]

    stage_names = {
        "url": "URL stage",
        "title_summary": "title+summary stage",
        "title": "title stage",
    }
    for stage_name, display_name in stage_names.items():
        stage = dedup_stats["stages"][stage_name]
        lines.extend(
            [
                f"- {display_name}: consistent_groups="
                f"{stage['consistent_groups']}, "
                f"conflicting_groups={stage['conflicting_groups']}, "
                f"rows_removed={stage['rows_removed']}",
            ]
        )

    lines.extend(["", "## Split", ""])
    lines.append("| Split | Rows | Start Date | End Date |")
    lines.append("| --- | ---: | --- | --- |")
    for name in ("train", "val", "test"):
        info = split_stats["splits"][name]
        lines.append(
            f"| {name.capitalize()} | {info['rows']} | "
            f"{info['start_date']} | {info['end_date']} |"
        )
    eval_info = split_stats["splits"]["eval_test"]
    lines.append(
        f"| eval_test | {eval_info['rows']} | "
        f"{eval_info['start_date']} | {eval_info['end_date']} |"
    )

    lines.extend(["", "## Label Distribution", ""])
    for name in ("train", "val", "test", "eval_test"):
        lines.append(f"### {name}")
        lines.append("| label | count |")
        lines.append("| ---: | ---: |")
        for label, count in sorted(
            split_label_dists[name].items(),
            key=lambda item: int(item[0]),
        ):
            lines.append(f"| {label} | {count} |")
        lines.append("")

    lines.extend(
        [
            "## Eval Quota",
            "",
            f"- eval_size: {eval_size}",
            f"- seed: {seed}",
            "- allocation: minimum allocation + remaining-capacity "
            "proportional allocation + largest remainder + label "
            "ascending tie-break",
        ]
    )
    for label in sorted(quotas):
        lines.append(f"- label {label}: {quotas[label]}")
    lines.append(f"- total: {sum(quotas.values())}")

    lines.extend(["", "## Integrity", ""])
    integrity_lines = [
        ("train_duplicate_sample_ids", "Train duplicate sample_id"),
        ("val_duplicate_sample_ids", "Val duplicate sample_id"),
        ("test_duplicate_sample_ids", "Test duplicate sample_id"),
        (
            "eval_test_duplicate_sample_ids",
            "eval_test duplicate sample_id",
        ),
        ("train_intersect_val", "Train ∩ Val"),
        ("train_intersect_test", "Train ∩ Test"),
        ("val_intersect_test", "Val ∩ Test"),
        ("eval_test_outside_test", "eval_test outside Test"),
    ]
    for key, display_name in integrity_lines:
        lines.append(f"- {display_name}: {integrity[key]}")

    lines.extend(["", "## Reproducibility", ""])
    if verification is None:
        lines.extend(
            [
                "- eval_test ordered sample_ids equal across two "
                "runs: not_yet_verified (second run pending)",
                "- eval_test CSV SHA256 equal across two runs: "
                "not_yet_verified (second run pending)",
            ]
        )
    else:
        eval_verification = verification["eval_test"]
        lines.append(
            "- eval_test ordered sample_ids equal across two runs: "
            f"{'True' if eval_verification['ordered_sample_ids_equal'] else 'False'}"
        )
        lines.append(
            "- eval_test CSV SHA256 equal across two runs: "
            f"{'True' if eval_verification['csv_sha256_equal'] else 'False'}"
        )
        for name in ("train", "val", "test"):
            csv_equal = verification[name]["csv_sha256_equal"]
            lines.append(
                f"- {name}.csv SHA256 equal across two runs: "
                f"{'True' if csv_equal else 'False'}"
            )

    lines.append("")
    return "\n".join(lines)


def compare_with_previous_run(
    output_dir: str | Path,
    verify_dir: str | Path,
) -> dict[str, dict[str, bool]]:
    """Compare generated split CSVs against a previous run's outputs."""
    results = {}
    for name in ("train", "val", "test", "eval_test"):
        current_csv = Path(output_dir) / f"{name}.csv"
        previous_csv = Path(verify_dir) / f"{name}.csv"
        results[name] = {
            "csv_sha256_equal": (
                sha256_file(current_csv) == sha256_file(previous_csv)
            ),
            "ordered_sample_ids_equal": (
                read_csv_sample_ids(current_csv)
                == read_csv_sample_ids(previous_csv)
            ),
        }
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal split freezing CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m preprocess.make_splits",
        description=(
            "Freeze reproducible time-based Risk splits plus a fixed "
            "deterministic eval_test."
        ),
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task name (currently only: risk)",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the source CSV",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that receives the split CSVs, manifest and report",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=500,
        help="Fixed eval_test size (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for eval sampling (default: 42)",
    )
    parser.add_argument(
        "--verify-dir",
        default=None,
        help=(
            "Optional directory holding a previous run's CSVs; "
            "CSV SHA256 and ordered sample ids are compared and the "
            "results are recorded in the report"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the frozen split pipeline end to end."""
    args = build_parser().parse_args(argv)

    label_column = TASK_LABEL_COLUMNS.get(args.task)
    if label_column is None:
        print(f"unsupported task: {args.task}", file=sys.stderr)
        return 2

    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(__file__).resolve().parent.parent

    records, clean_stats = load_and_clean_csv(
        source_path,
        label_column=label_column,
    )
    deduplicated, dedup_stats = deduplicate_records(records)
    del records

    splits, split_stats = split_records_by_time(deduplicated)

    eval_records, quotas = sample_eval_records(
        splits["test"],
        eval_size=args.eval_size,
        labels=EVAL_LABELS,
        seed=args.seed,
    )

    for name in ("train", "val", "test"):
        write_split_csv(output_dir / f"{name}.csv", splits[name])
    write_split_csv(output_dir / "eval_test.csv", eval_records)

    eval_dates = [
        record["date"] for record in eval_records
    ]
    split_stats["splits"]["eval_test"] = {
        "rows": len(eval_records),
        "start_date": min(eval_dates).isoformat(),
        "end_date": max(eval_dates).isoformat(),
    }

    integrity = check_split_integrity(splits, eval_records)
    verification = (
        compare_with_previous_run(output_dir, args.verify_dir)
        if args.verify_dir
        else None
    )

    source_physical_lines = count_physical_lines(source_path)
    manifest = build_manifest(
        task=args.task,
        source_path=source_path,
        source_rows=clean_stats["raw_rows"],
        source_physical_lines=source_physical_lines,
        eval_size=args.eval_size,
        seed=args.seed,
        output_dir=output_dir,
        splits=splits,
        eval_records=eval_records,
        cutoffs=split_stats["cutoffs"],
        repo_dir=repo_dir,
    )
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = build_report(
        task=args.task,
        source_path=source_path,
        source_sha256=sha256_file(source_path),
        source_physical_lines=source_physical_lines,
        clean_stats=clean_stats,
        dedup_stats=dedup_stats,
        split_stats=split_stats,
        eval_size=args.eval_size,
        seed=args.seed,
        quotas=quotas,
        split_label_dists={
            name: label_distribution(
                eval_records if name == "eval_test" else splits[name]
            )
            for name in ("train", "val", "test", "eval_test")
        },
        integrity=integrity,
        verification=verification,
    )
    report_path = output_dir / "split_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"task: {args.task}")
    print(f"source logical csv rows: {clean_stats['raw_rows']}")
    print(
        "source physical text lines including header: "
        f"{source_physical_lines}"
    )
    print(f"after cleaning: {clean_stats['rows_after_cleaning']}")
    print(
        f"unique removed rows: {clean_stats['unique_removed_rows']}, "
        f"reason hits: {clean_stats['reason_hits']}, "
        f"overlap: {clean_stats['overlap']}"
    )
    print(f"after dedup: {dedup_stats['output_rows']}")
    print(
        "dedup removed: "
        f"url={dedup_stats['stages']['url']['rows_removed']}, "
        "title_summary="
        f"{dedup_stats['stages']['title_summary']['rows_removed']}, "
        f"title={dedup_stats['stages']['title']['rows_removed']}"
    )
    print(
        "split rows: "
        + ", ".join(
            f"{name}={split_stats['splits'][name]['rows']}"
            for name in ("train", "val", "test", "eval_test")
        )
    )
    print(
        "eval quotas: "
        + ", ".join(
            f"label {label}={quotas[label]}"
            for label in sorted(quotas)
        )
        + f", total={sum(quotas.values())}"
    )
    print(
        "integrity: "
        + ", ".join(
            f"{key}={value}" for key, value in integrity.items()
        )
    )
    if verification is not None:
        for name, results in verification.items():
            print(
                f"verify {name}: csv_sha256_equal="
                f"{results['csv_sha256_equal']}, "
                f"ordered_sample_ids_equal="
                f"{results['ordered_sample_ids_equal']}"
            )
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
