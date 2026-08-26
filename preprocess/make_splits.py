"""Build reproducible time-based dataset splits."""

import hashlib
import math
import unicodedata
from datetime import date, datetime, timezone


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
