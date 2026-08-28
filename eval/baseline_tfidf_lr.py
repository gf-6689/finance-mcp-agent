"""TF-IDF + Logistic Regression baseline for the frozen Risk splits.

The pipeline is frozen by the 3-day baseline plan:

- features: title + summary only (title.strip() + "\\n" + summary.strip())
- TF-IDF: ngram_range=(1, 2), max_features=50000, fit on Train only
- LR: class_weight="balanced", max_iter=1000; only C in {0.5, 1.0, 2.0}
- C is selected solely by Val Macro-F1; ties pick the smallest C
- the fixed 500-row eval_test is used once, for the frozen config only
- latency uses time.perf_counter around TF-IDF transform + LR predict
"""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from eval.metrics import RISK_LABELS, accuracy, macro_f1
from preprocess.make_splits import (
    git_commit_sha,
    git_working_tree_dirty,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits" / "risk"
C_CANDIDATES = (0.5, 1.0, 2.0)
MODEL_NAME = "tfidf_logistic_regression"

FROZEN_EVAL_TEST_SHA256 = (
    "b052920047da97cc3e7e7ab4df382e79ebc9ab49e4fd466e7364807972272986"
)

# Validated sequential blue ramp (light -> dark) for magnitude cells.
BLUE_RAMP = {
    100: "#cde2fb",
    150: "#b7d3f6",
    200: "#9ec5f4",
    250: "#86b6ef",
    300: "#6da7ec",
    350: "#5598e7",
    400: "#3987e5",
    450: "#2a78d6",
    500: "#256abf",
    550: "#1c5cab",
    600: "#184f95",
    650: "#104281",
    700: "#0d366b",
}


def load_split_texts_and_labels(
    path: Path,
) -> tuple[list[str], list[str], list[int]]:
    """Load sample ids, title+summary texts and labels in file order."""
    sample_ids: list[str] = []
    texts: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_ids.append(row["sample_id"])
            texts.append(
                row["title"].strip() + "\n" + row["summary"].strip()
            )
            labels.append(int(row["label"]))
    return sample_ids, texts, labels


def verify_frozen_splits(manifest: dict) -> None:
    """Fail loudly when any split CSV differs from the frozen evidence."""
    for name in ("train", "val", "test", "eval_test"):
        expected = manifest["splits"][name]["csv_sha256"]
        actual = sha256_file(SPLITS_DIR / f"{name}.csv")
        if actual != expected:
            raise RuntimeError(
                f"{name}.csv SHA256 mismatch: "
                f"manifest={expected} actual={actual}"
            )

    eval_manifest_sha = manifest["splits"]["eval_test"]["csv_sha256"]
    if eval_manifest_sha != FROZEN_EVAL_TEST_SHA256:
        raise RuntimeError(
            "eval_test.csv SHA256 differs from the frozen "
            "checkpoint value"
        )


def write_confusion_matrix_png(
    y_true: list[int],
    y_pred: list[int],
    path: Path,
) -> None:
    """Render the fixed-order 1..5 confusion matrix as a PNG.

    Diagnostic only: it never influences parameter selection.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=RISK_LABELS)
    cmap = LinearSegmentedColormap.from_list(
        "blue-ramp",
        [BLUE_RAMP[step] for step in sorted(BLUE_RAMP)],
    )

    figure, axis = plt.subplots(figsize=(6.0, 5.2))
    image = axis.imshow(matrix, cmap=cmap)

    threshold = matrix.max() * 0.45
    for row_index in range(len(RISK_LABELS)):
        for column_index in range(len(RISK_LABELS)):
            value = int(matrix[row_index, column_index])
            text_color = (
                "white" if value > threshold else "#1a1a19"
            )
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    axis.set_xticks(range(len(RISK_LABELS)), RISK_LABELS)
    axis.set_yticks(range(len(RISK_LABELS)), RISK_LABELS)
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Gold")
    axis.set_title(
        "Risk TF-IDF + LR Confusion Matrix (eval_test, N=500)"
    )
    figure.colorbar(image, ax=axis, label="count")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run(output_dir: Path) -> None:
    """Run the frozen TF-IDF + LR pipeline and write all artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = SPLITS_DIR / "split_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    verify_frozen_splits(manifest)

    _train_ids, train_texts, train_labels = (
        load_split_texts_and_labels(SPLITS_DIR / "train.csv")
    )
    _val_ids, val_texts, val_labels = load_split_texts_and_labels(
        SPLITS_DIR / "val.csv"
    )
    eval_ids, eval_texts, eval_labels = load_split_texts_and_labels(
        SPLITS_DIR / "eval_test.csv"
    )

    if len(eval_labels) != 500:
        raise RuntimeError(
            f"eval_test rows must be 500, got {len(eval_labels)}"
        )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)

    val_search = []
    for c_value in C_CANDIDATES:
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=1000,
        )
        model.fit(x_train, train_labels)
        predictions = model.predict(x_val)
        val_search.append(
            {
                "C": c_value,
                "val_macro_f1": macro_f1(val_labels, predictions),
            }
        )

    best_f1 = max(
        entry["val_macro_f1"] for entry in val_search
    )
    selected_c = min(
        entry["C"]
        for entry in val_search
        if entry["val_macro_f1"] == best_f1
    )

    frozen_model = LogisticRegression(
        C=selected_c,
        class_weight="balanced",
        max_iter=1000,
    )
    frozen_model.fit(x_train, train_labels)

    start = time.perf_counter()
    x_eval = vectorizer.transform(eval_texts)
    predictions = frozen_model.predict(x_eval)
    total_inference_seconds = time.perf_counter() - start
    mean_latency_ms_per_sample = (
        total_inference_seconds / len(eval_texts) * 1000
    )

    prediction_labels = [int(value) for value in predictions]
    eval_macro_f1 = macro_f1(eval_labels, prediction_labels)
    eval_accuracy = accuracy(eval_labels, prediction_labels)

    with (output_dir / "risk_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_id", "gold_label", "prediction"])
        for sample_id, gold_label, prediction in zip(
            eval_ids, eval_labels, prediction_labels
        ):
            writer.writerow([sample_id, gold_label, prediction])

    (output_dir / "risk_val_search.json").write_text(
        json.dumps(
            {
                "task": "risk",
                "model": MODEL_NAME,
                "val_search": val_search,
                "selected_C": selected_c,
                "selection_rule": (
                    "choose C with the highest Val Macro-F1; "
                    "ties are broken by the smallest C"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (output_dir / "risk_metrics.json").write_text(
        json.dumps(
            {
                "task": "risk",
                "model": MODEL_NAME,
                "train_rows": len(train_labels),
                "val_rows": len(val_labels),
                "eval_rows": len(eval_labels),
                "selected_C": selected_c,
                "train_sha256": manifest["splits"]["train"][
                    "csv_sha256"
                ],
                "val_sha256": manifest["splits"]["val"][
                    "csv_sha256"
                ],
                "eval_test_sha256": manifest["splits"]["eval_test"][
                    "csv_sha256"
                ],
                "macro_f1": eval_macro_f1,
                "accuracy": eval_accuracy,
                "total_inference_seconds": total_inference_seconds,
                "mean_latency_ms_per_sample": (
                    mean_latency_ms_per_sample
                ),
                "gold_label_distribution": {
                    str(label): eval_labels.count(label)
                    for label in RISK_LABELS
                },
                "prediction_label_distribution": {
                    str(label): prediction_labels.count(label)
                    for label in RISK_LABELS
                },
                "tfidf_ngram_range": [1, 2],
                "tfidf_max_features": 50000,
                "lr_class_weight": "balanced",
                "lr_max_iter": 1000,
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
                "git_commit": git_commit_sha(REPO_ROOT),
                "working_tree_dirty": git_working_tree_dirty(REPO_ROOT),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    write_confusion_matrix_png(
        eval_labels,
        prediction_labels,
        output_dir / "risk_confusion_matrix.png",
    )

    print("val search:")
    for entry in val_search:
        print(
            f"  C={entry['C']}  val_macro_f1={entry['val_macro_f1']:.6f}"
        )
    print(f"selected_C: {selected_c}")
    print(f"eval Macro-F1: {eval_macro_f1:.6f}")
    print(f"eval Accuracy: {eval_accuracy:.6f}")
    print(
        f"total_inference_seconds: {total_inference_seconds:.6f}"
    )
    print(
        "mean_latency_ms_per_sample: "
        f"{mean_latency_ms_per_sample:.6f}"
    )
    print(
        "prediction distribution: "
        + json.dumps(
            {
                str(label): prediction_labels.count(label)
                for label in RISK_LABELS
            }
        )
    )
    print(f"outputs written to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.baseline_tfidf_lr",
        description=(
            "Frozen TF-IDF + Logistic Regression baseline over the "
            "Risk splits."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that receives all result artifacts",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
