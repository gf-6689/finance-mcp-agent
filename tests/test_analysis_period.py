from datetime import date
from pathlib import Path
import sys

import pytest


MCP_ROOT = Path(__file__).resolve().parents[1] / "a-share-mcp-is-just-i-need"
sys.path.insert(0, str(MCP_ROOT))

from src.tools.analysis import latest_disclosed_financial_period

for module_name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
    sys.modules.pop(module_name)


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        (date(2026, 1, 1), (2025, 3)),
        (date(2026, 4, 30), (2025, 3)),
        (date(2026, 5, 1), (2026, 1)),
        (date(2026, 8, 16), (2026, 1)),
        (date(2026, 9, 1), (2026, 2)),
        (date(2026, 10, 31), (2026, 2)),
        (date(2026, 11, 1), (2026, 3)),
        (date(2026, 12, 31), (2026, 3)),
    ],
)
def test_latest_disclosed_financial_period_uses_reporting_deadlines(as_of, expected):
    assert latest_disclosed_financial_period(as_of) == expected
