import pytest

from eval.metrics import RISK_LABELS, accuracy, macro_f1


def test_risk_labels_are_frozen_to_one_through_five():
    assert RISK_LABELS == [1, 2, 3, 4, 5]


def test_accuracy_is_correct():
    assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert accuracy([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)


def test_macro_f1_is_one_for_perfect_predictions():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 4, 5]

    assert macro_f1(y_true, y_pred) == 1.0


def test_macro_f1_keeps_all_five_labels_when_only_two_appear():
    # Perfect on two classes, but classes 3/4/5 have zero support:
    # Macro-F1 must be (1 + 1 + 0 + 0 + 0) / 5, not 1.0.
    y_true = [1, 1, 2, 2]
    y_pred = [1, 1, 2, 2]

    assert macro_f1(y_true, y_pred) == pytest.approx(0.4)


def test_macro_f1_handles_a_class_without_predictions():
    # Class 1: precision 0.2, recall 1.0 -> F1 1/3. Classes 2-5 have
    # no predictions at all and contribute zero each.
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 1, 1, 1, 1]

    assert macro_f1(y_true, y_pred) == pytest.approx(
        (1 / 3) / 5
    )


def test_macro_f1_zero_division_contributes_zero():
    # Classes with no true positives, no false positives and no false
    # negatives divide by zero; zero_division=0 makes them zero.
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 1, 2]

    expected = pytest.approx(0.2)
    assert macro_f1(y_true, y_pred) == expected


def test_accuracy_and_macro_f1_raise_on_length_mismatch():
    y_true = [1, 2, 3]
    y_pred = [1, 2]

    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        accuracy(y_true, y_pred)

    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        macro_f1(y_true, y_pred)
