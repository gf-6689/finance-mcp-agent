"""Unified evaluation metrics for the Risk baseline comparison."""

from sklearn.metrics import accuracy_score, f1_score

RISK_LABELS = [1, 2, 3, 4, 5]


def _check_lengths(y_true, y_pred) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length: "
            f"{len(y_true)} != {len(y_pred)}"
        )


def accuracy(y_true, y_pred) -> float:
    """Compute accuracy over the given gold and predicted labels."""
    _check_lengths(y_true, y_pred)
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred) -> float:
    """Compute Macro-F1 over the frozen five Risk labels.

    The label set is always [1, 2, 3, 4, 5], never shrunk to the
    classes present in y_true or y_pred: a class with zero support
    contributes a zero-division zero instead of being dropped.
    """
    _check_lengths(y_true, y_pred)
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=RISK_LABELS,
            average="macro",
            zero_division=0,
        )
    )
