"""Deterministic sampling logic for the formal QLoRA training freeze.

Pure, testable logic only: no GPU, no CSV I/O of the frozen splits.

Frozen rules (train-size sampling, Checkpoint 4A):

- sample without replacement from the frozen train.csv only;
- seed = 42;
- labels listed in always_keep_labels are kept in full (label 5 has
  only 36 rows in Train and is always fully retained, never copied);
- the remaining seats are shared over the other labels
  proportionally to their original Train counts, floors first, then
  largest remainder, ties broken by ascending label;
- every quota is capped by the label's real count;
- the final ordered sample id list must hash to a fixed SHA256.
"""

import math
import random
from collections import Counter
from collections.abc import Mapping


def compute_train_quotas(
    counts: Mapping[int, int],
    *,
    train_size: int,
    always_keep_labels: frozenset[int],
) -> dict[int, int]:
    """Compute deterministic per-label quotas for the formal train set.

    Labels in always_keep_labels receive their full count first. The
    remaining seats are allocated proportionally to the original
    counts of the other labels with the largest remainder method and
    ascending-label tie-break. Quotas never exceed real counts and
    always sum to exactly train_size.
    """
    if train_size < 0:
        raise ValueError("train_size must be non-negative")

    present = sorted(label for label in counts if counts[label] > 0)
    total_available = sum(counts[label] for label in present)
    if total_available < train_size:
        raise ValueError(
            "Training set has fewer rows than train_size: "
            f"{total_available} < {train_size}"
        )

    quota = {
        label: counts[label]
        for label in present
        if label in always_keep_labels
    }
    remaining = train_size - sum(quota.values())
    if remaining < 0:
        raise ValueError(
            "always_keep_labels exceed train_size"
        )

    eligible = [
        label for label in present if label not in always_keep_labels
    ]
    if remaining == 0:
        return quota

    total_weight = sum(counts[label] for label in eligible)
    if total_weight <= 0:
        raise ValueError(
            "No eligible labels left for the remaining seats"
        )

    ideal = {
        label: remaining * counts[label] / total_weight
        for label in eligible
    }
    floor_extra = {
        label: math.floor(ideal[label]) for label in eligible
    }
    for label in eligible:
        addition = min(
            floor_extra[label],
            counts[label] - quota.get(label, 0),
        )
        quota[label] = quota.get(label, 0) + addition

    remaining = train_size - sum(quota.values())
    if remaining > 0:
        order = sorted(
            eligible,
            key=lambda label: (
                -(ideal[label] - floor_extra[label]),
                label,
            ),
        )
        progressed = True
        while remaining > 0 and progressed:
            progressed = False
            for label in order:
                if remaining == 0:
                    break
                if quota[label] < counts[label]:
                    quota[label] += 1
                    remaining -= 1
                    progressed = True
        if remaining > 0:
            raise ValueError(
                "Cannot reach train_size: every label at capacity"
            )

    return quota


def sample_train_records(
    records: list[dict[str, object]],
    *,
    train_size: int,
    always_keep_labels: frozenset[int],
    seed: int,
) -> tuple[list[dict[str, object]], dict[int, int]]:
    """Draw the deterministic formal train subset from train records.

    Pools are ordered by sample id before the seeded draw so results
    never depend on input order, and the returned records are sorted
    by sample id ascending so the output row order is fixed.
    """
    counts: Counter[int] = Counter()
    for record in records:
        counts[int(record["label"])] += 1

    quotas = compute_train_quotas(
        counts,
        train_size=train_size,
        always_keep_labels=always_keep_labels,
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
                for record in records
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
