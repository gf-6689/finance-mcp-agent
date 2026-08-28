"""Strict parser for the frozen FINAL_LABEL=X output protocol.

Frozen rules (version 1):

- the raw output must contain exactly one FINAL_LABEL=<integer>
  assignment;
- the integer must be in [1, 5];
- zero assignments, multiple assignments (consistent or not), and
  out-of-range or malformed values are all invalid;
- other digits in the text never rescue an invalid output;
- an invalid output yields prediction = -1, which is an internal
  sentinel only and always counts as a wrong answer in the metrics.

These rules are fixed; they may not be tuned against eval_test output.
"""

import re

PARSER_VERSION = "1"

VALID_LABELS = frozenset((1, 2, 3, 4, 5))

# Matches a FINAL_LABEL=<digits> assignment whose value does not
# continue with another digit or a decimal point, so "FINAL_LABEL=3.0"
# and "FINAL_LABEL=30" do not parse as the integer 3.
_ASSIGNMENT_RE = re.compile(r"FINAL_LABEL=(\d+)(?![.\d])")


def parse_final_label(raw_output) -> tuple[int, bool]:
    """Parse exactly one FINAL_LABEL=X (X in 1..5) from raw output.

    Returns (prediction, valid_output). Invalid inputs return
    (-1, False).
    """
    if not isinstance(raw_output, str):
        return -1, False

    matches = _ASSIGNMENT_RE.findall(raw_output)
    if len(matches) != 1:
        return -1, False

    value = int(matches[0])
    if value not in VALID_LABELS:
        return -1, False

    return value, True
