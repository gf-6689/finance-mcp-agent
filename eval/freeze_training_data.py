"""Freeze the formal QLoRA training sample lists (Checkpoint 4A).

Deterministic, offline, never touches eval_test:

- formal train set: 20,000 rows sampled from train.csv (seed 42,
  without replacement, all 36 label-5 rows kept, the rest split by
  original Train proportions with the largest remainder method);
- Val checkpoint-selection subset: 1,000 rows sampled from val.csv
  (seed 42) with the same deterministic stratified allocation used
  for the eval_test quotas.

Both sample id lists are frozen to files plus a manifest with the
source CSV SHA256 and the sample_ids SHA256.
"""

import argparse
import csv
import json
from pathlib import Path

from eval.train_sampling import sample_train_records
from preprocess.make_splits import (
    EVAL_LABELS,
    sample_eval_records,
    sha256_file,
    sha256_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits" / "risk"
TRAINING_DIR = REPO_ROOT / "data" / "training"

FORMAL_TRAIN_SIZE = 20000
TRAIN_ALWAYS_KEEP_LABELS = frozenset({5})
VAL_SELECTION_SIZE = 1000
SEED = 42


def load_records(csv_path: Path) -> list[dict]:
    records = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "label": int(row["label"]),
                }
            )
    return records


def write_sample_ids(path: Path, sample_ids: list[str]) -> None:
    path.write_text(
        "\n".join(sample_ids) + "\n",
        encoding="utf-8",
    )


def label_distribution(records: list[dict]) -> dict[str, int]:
    return {
        str(label): sum(
            1 for record in records if record["label"] == label
        )
        for label in EVAL_LABELS
    }


def run(training_dir: Path) -> None:
    training_dir.mkdir(parents=True, exist_ok=True)

    # --- formal train set -------------------------------------------------
    train_path = SPLITS_DIR / "train.csv"
    train_records = load_records(train_path)

    sampled, quotas = sample_train_records(
        train_records,
        train_size=FORMAL_TRAIN_SIZE,
        always_keep_labels=TRAIN_ALWAYS_KEEP_LABELS,
        seed=SEED,
    )
    train_ids = [record["sample_id"] for record in sampled]
    assert len(train_ids) == FORMAL_TRAIN_SIZE
    assert len(set(train_ids)) == FORMAL_TRAIN_SIZE

    train_ids_path = (
        training_dir / "risk_qlora_train_sample_ids.txt"
    )
    write_sample_ids(train_ids_path, train_ids)
    train_ids_sha256 = sha256_text("\n".join(train_ids))

    train_manifest = {
        "purpose": "formal QLoRA training set (frozen)",
        "source": {
            "path": str(train_path.relative_to(REPO_ROOT)),
            "csv_sha256": sha256_file(train_path),
        },
        "formal_train_size": FORMAL_TRAIN_SIZE,
        "seed": SEED,
        "sampling_method": (
            "without replacement; label 5 kept in full; remaining "
            "seats proportional to original Train label counts with "
            "largest remainder and ascending-label tie-break"
        ),
        "always_keep_labels": sorted(TRAIN_ALWAYS_KEEP_LABELS),
        "quotas": {
            str(label): quotas[label] for label in sorted(quotas)
        },
        "label_distribution": label_distribution(sampled),
        "sample_ids_file": (
            str(train_ids_path.relative_to(REPO_ROOT))
        ),
        "sample_ids_sha256": train_ids_sha256,
    }
    (training_dir / "risk_qlora_train_manifest.json").write_text(
        json.dumps(train_manifest, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    # --- Val checkpoint-selection subset ---------------------------------
    val_path = SPLITS_DIR / "val.csv"
    val_records = load_records(val_path)

    val_subset, val_quotas = sample_eval_records(
        val_records,
        eval_size=VAL_SELECTION_SIZE,
        labels=EVAL_LABELS,
        seed=SEED,
    )
    val_ids = [record["sample_id"] for record in val_subset]
    assert len(val_ids) == VAL_SELECTION_SIZE
    assert len(set(val_ids)) == VAL_SELECTION_SIZE

    val_ids_path = training_dir / (
        "risk_qlora_val_checkpoint_selection_sample_ids.txt"
    )
    write_sample_ids(val_ids_path, val_ids)
    val_ids_sha256 = sha256_text("\n".join(val_ids))

    val_manifest = {
        "purpose": (
            "Val subset for QLoRA checkpoint selection (frozen); "
            "never used for training"
        ),
        "source": {
            "path": str(val_path.relative_to(REPO_ROOT)),
            "csv_sha256": sha256_file(val_path),
        },
        "subset_size": VAL_SELECTION_SIZE,
        "seed": SEED,
        "sampling_method": (
            "same deterministic stratified allocation as eval_test "
            "quotas: class minimum 10 then remaining-capacity "
            "largest remainder, ascending-label tie-break"
        ),
        "quotas": {
            str(label): val_quotas[label]
            for label in sorted(val_quotas)
        },
        "label_distribution": label_distribution(val_subset),
        "sample_ids_file": (
            str(val_ids_path.relative_to(REPO_ROOT))
        ),
        "sample_ids_sha256": val_ids_sha256,
    }
    (training_dir / (
        "risk_qlora_val_checkpoint_selection_manifest.json"
    )).write_text(
        json.dumps(val_manifest, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    print(f"train sampled rows: {len(train_ids)}")
    print(f"train quotas: {json.dumps(train_manifest['quotas'])}")
    print(f"train label distribution: "
          f"{json.dumps(train_manifest['label_distribution'])}")
    print(f"train sample_ids_sha256: {train_ids_sha256}")
    print(f"val subset rows: {len(val_ids)}")
    print(f"val quotas: {json.dumps(val_manifest['quotas'])}")
    print(f"val sample_ids_sha256: {val_ids_sha256}")
    print(f"manifests written to: {training_dir}")
    print("used eval_test: False")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.freeze_training_data",
        description=(
            "Freeze the formal QLoRA training sample lists and "
            "manifests."
        ),
    )
    parser.add_argument(
        "--training-dir",
        default=str(TRAINING_DIR),
        help="Directory that receives the frozen sample id lists",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run(Path(args.training_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
