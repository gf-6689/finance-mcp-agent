from datetime import date

import pytest

from eval.train_sampling import (
    compute_train_quotas,
    sample_train_records,
)

KEEP_5 = frozenset({5})


def _make_train_records(counts):
    """Build unique synthetic train records with the given counts."""
    records = []
    row_id = 0
    for label in sorted(counts):
        for index in range(counts[label]):
            records.append(
                {
                    "sample_id": f"l{label}-{index:05d}",
                    "date": date(2023, 1, 1),
                    "title": f"title {label} {index}",
                    "summary": f"summary {label} {index}",
                    "stock_symbol": "TEST",
                    "url": f"https://example.com/{label}/{index}",
                    "label": label,
                    "_original_row_id": row_id,
                }
            )
            row_id += 1
    return records


def test_compute_train_quotas_keeps_label_5_in_full():
    counts = {1: 291, 2: 6672, 3: 39095, 4: 4494, 5: 36}

    quotas = compute_train_quotas(
        counts,
        train_size=20000,
        always_keep_labels=KEEP_5,
    )

    assert quotas[5] == 36
    assert sum(quotas.values()) == 20000
    for label, count in counts.items():
        assert quotas[label] <= count


def test_compute_train_quotas_matches_frozen_expected_values():
    # Hand-verified expected values for the frozen Train distribution
    # and train_size 20000: label 5 kept in full, the remaining 19964
    # seats split proportionally over labels 1..4 (ideals 114.9218,
    # 2634.9068, 15439.4006, 1774.7709 -> floors sum to 19961), then
    # the 3 leftover seats go to the largest remainders: labels
    # 1, 2 and 4 each receive one.
    counts = {1: 291, 2: 6672, 3: 39095, 4: 4494, 5: 36}

    quotas = compute_train_quotas(
        counts,
        train_size=20000,
        always_keep_labels=KEEP_5,
    )

    assert quotas == {1: 115, 2: 2635, 3: 15439, 4: 1775, 5: 36}


def test_compute_train_quotas_breaks_ties_by_label_ascending():
    counts = {1: 100, 2: 100, 3: 20}
    quotas = compute_train_quotas(
        counts,
        train_size=111,
        always_keep_labels=frozenset({3}),
    )

    assert quotas[3] == 20
    # 91 seats over equal capacities leave one seat with identical
    # remainders: label 1 wins the tie.
    assert quotas[1] == 46
    assert quotas[2] == 45


def test_compute_train_quotas_raises_when_train_too_small():
    with pytest.raises(
        ValueError,
        match="fewer rows than train_size",
    ):
        compute_train_quotas(
            {1: 50, 2: 60},
            train_size=200,
            always_keep_labels=KEEP_5,
        )


def test_sample_train_records_reaches_target_without_duplicates():
    records = _make_train_records(
        {1: 291, 2: 6672, 3: 39095, 4: 4494, 5: 36}
    )

    sampled, quotas = sample_train_records(
        records,
        train_size=20000,
        always_keep_labels=KEEP_5,
        seed=42,
    )

    sample_ids = [record["sample_id"] for record in sampled]
    assert len(sample_ids) == 20000
    assert len(set(sample_ids)) == 20000
    assert len(
        [record for record in sampled if record["label"] == 5]
    ) == 36
    for label, quota in quotas.items():
        assert (
            sum(1 for record in sampled if record["label"] == label)
            == quota
        )


def test_sample_train_records_is_reproducible_and_order_independent():
    records = _make_train_records({1: 100, 2: 100, 3: 60, 5: 30})
    reversed_records = list(reversed(records))

    first, _first_quotas = sample_train_records(
        records,
        train_size=200,
        always_keep_labels=KEEP_5,
        seed=42,
    )
    second, _second_quotas = sample_train_records(
        reversed_records,
        train_size=200,
        always_keep_labels=KEEP_5,
        seed=42,
    )

    first_ids = [record["sample_id"] for record in first]
    second_ids = [record["sample_id"] for record in second]
    assert first_ids == second_ids
    assert first_ids == sorted(first_ids)
