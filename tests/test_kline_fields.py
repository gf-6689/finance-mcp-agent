from pathlib import Path
import sys

import pytest


MCP_ROOT = Path(__file__).resolve().parents[1] / "a-share-mcp-is-just-i-need"
sys.path.insert(0, str(MCP_ROOT))

from src.baostock_data_source import resolve_k_fields


DAILY_ONLY_FIELDS = {
    "preclose",
    "turn",
    "tradestatus",
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "isST",
}


def test_daily_defaults_keep_daily_and_valuation_fields():
    resolved = set(resolve_k_fields("d", None).split(","))

    assert {"date", "close", "preclose", "peTTM"} <= resolved


@pytest.mark.parametrize("frequency", ["w", "m"])
def test_aggregate_defaults_exclude_daily_only_fields(frequency):
    resolved = set(resolve_k_fields(frequency, None).split(","))

    assert {"date", "code", "open", "high", "low", "close"} <= resolved
    assert resolved.isdisjoint(DAILY_ONLY_FIELDS)


def test_valid_explicit_monthly_fields_are_preserved():
    assert resolve_k_fields("m", ["date", "code", "close"]) == "date,code,close"


def test_invalid_explicit_monthly_fields_fail_locally():
    with pytest.raises(ValueError, match=r"frequency 'm'.*peTTM"):
        resolve_k_fields("m", ["date", "close", "peTTM"])
