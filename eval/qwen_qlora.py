"""Formal Risk QLoRA checkpoint selection and evaluation (Checkpoint 4B).

Frozen 4B protocol:

- training: eval/train_qlora_risk.py under config/risk_qlora_training.json;
  the Trainer internal best-model selection is fully disabled and one
  checkpoint is saved per epoch (all three kept);
- checkpoint selection: post-training generative eval over the frozen
  Val1000 subset with Prompt v1 / Parser v1; the highest Val Macro-F1
  wins, an exact tie breaks toward the smaller global_step;
- the selected adapter is frozen to selected_adapter_manifest.json
  (adapter files + SHA256); eval_test may be read only after that
  manifest exists and verifies;
- formal eval: fixed 500-row eval_test, BF16 base + selected adapter,
  generation protocol identical to the frozen Base eval.
"""

import argparse
import csv
import gc
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.metrics import RISK_LABELS, accuracy, macro_f1
from eval.parser import PARSER_VERSION, parse_final_label
from eval.prompt import RISK_PROMPT_VERSION
from eval.qwen_base import (
    FROZEN_EVAL_TEST_SHA256,
    generate_one,
    load_samples,
    verify_frozen_eval_test,
)
from preprocess.make_splits import (
    git_commit_sha,
    git_working_tree_dirty,
    sha256_file,
    sha256_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits" / "risk"
TRAINING_DIR = REPO_ROOT / "data" / "training"
MODEL_PATH = REPO_ROOT / "Qwen"

CONFIG_PATH = REPO_ROOT / "config" / "risk_qlora_training.json"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "risk_qlora_formal_v1"
RESULTS_DIR = REPO_ROOT / "results" / "qwen" / "qlora"
SELECTED_ADAPTER_MANIFEST = (
    RESULTS_DIR / "selected_adapter_manifest.json"
)
VAL_SELECTION_FILE = RESULTS_DIR / "val_checkpoint_selection.json"

# Frozen at Checkpoint 4A and pinned here independently of the config
# so the training/eval code fails loudly if either list ever changes.
FROZEN_TRAIN_IDS_SHA256 = (
    "e975eb686e33b7614076e8e233321592224df25e4686a51cf3650baa3b538a0b"
)
FROZEN_VAL_IDS_SHA256 = (
    "98d857146c51e522db1eb3ce789d116d46e270753d6ff373523f5853ed9ffe9e"
)


def load_frozen_config() -> dict:
    """Load the frozen formal training/eval config."""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _resolve(repo_root: Path, value: str) -> Path:
    """Resolve a manifest-recorded path (absolute, or repo-relative)."""
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_sample_ids(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _load_csv_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["sample_id"]] = {
                "sample_id": row["sample_id"],
                "label": int(row["label"]),
                "title": row["title"],
                "summary": row["summary"],
            }
    return rows


def _label_distribution(sample_ids, row_map) -> dict[str, int]:
    return {
        str(label): sum(
            1
            for sample_id in sample_ids
            if row_map[sample_id]["label"] == label
        )
        for label in RISK_LABELS
    }


def verify_frozen_training_data(
    config: dict,
    repo_root: Path = REPO_ROOT,
    training_dir: Path = TRAINING_DIR,
    splits_dir: Path = SPLITS_DIR,
    expected_train_ids_sha256: str | None = FROZEN_TRAIN_IDS_SHA256,
    expected_val_ids_sha256: str | None = FROZEN_VAL_IDS_SHA256,
) -> dict:
    """Run the nine frozen-data checks before formal training.

    Never reads test.csv or eval_test.csv sample content.
    """
    split_manifest = json.loads(
        (splits_dir / "split_manifest.json").read_text(encoding="utf-8")
    )

    # 1/2: source CSV SHA256 must match the frozen split manifest.
    for name in ("train", "val"):
        csv_path = splits_dir / f"{name}.csv"
        expected = split_manifest["splits"][name]["csv_sha256"]
        actual = sha256_file(csv_path)
        if actual != expected:
            raise RuntimeError(
                f"{name}.csv SHA256 mismatch: "
                f"manifest={expected} actual={actual}"
            )

    train_manifest_path = _resolve(repo_root, config["train_manifest"])
    val_manifest_path = _resolve(
        repo_root, config["checkpoint_selection"]["eval_subset_manifest"]
    )
    train_manifest = json.loads(
        train_manifest_path.read_text(encoding="utf-8")
    )
    val_manifest = json.loads(val_manifest_path.read_text(encoding="utf-8"))

    train_ids_path = _resolve(
        repo_root, train_manifest["sample_ids_file"]
    )
    val_ids_path = _resolve(repo_root, val_manifest["sample_ids_file"])

    train_ids = _load_sample_ids(train_ids_path)
    val_ids = _load_sample_ids(val_ids_path)

    # 3/4: sample-id list SHA256 must match the manifest and the frozen
    # checkpoint constants.
    train_ids_sha256 = sha256_text("\n".join(train_ids))
    val_ids_sha256 = sha256_text("\n".join(val_ids))
    if train_ids_sha256 != train_manifest["sample_ids_sha256"]:
        raise RuntimeError(
            "train sample_ids SHA256 mismatch: "
            f"manifest={train_manifest['sample_ids_sha256']} "
            f"actual={train_ids_sha256}"
        )
    if train_ids_sha256 != config["train_sample_ids_sha256"]:
        raise RuntimeError(
            "train sample_ids SHA256 mismatch vs frozen config: "
            f"config={config['train_sample_ids_sha256']} "
            f"actual={train_ids_sha256}"
        )
    if expected_train_ids_sha256 is not None and (
        train_ids_sha256 != expected_train_ids_sha256
    ):
        raise RuntimeError(
            "train sample_ids SHA256 mismatch vs 4A constant: "
            f"constant={expected_train_ids_sha256} "
            f"actual={train_ids_sha256}"
        )
    if val_ids_sha256 != val_manifest["sample_ids_sha256"]:
        raise RuntimeError(
            "val sample_ids SHA256 mismatch: "
            f"manifest={val_manifest['sample_ids_sha256']} "
            f"actual={val_ids_sha256}"
        )
    if expected_val_ids_sha256 is not None and (
        val_ids_sha256 != expected_val_ids_sha256
    ):
        raise RuntimeError(
            "val sample_ids SHA256 mismatch vs 4A constant: "
            f"constant={expected_val_ids_sha256} "
            f"actual={val_ids_sha256}"
        )

    if len(train_ids) != config["train_size"]:
        raise RuntimeError(
            f"train sample ids count {len(train_ids)} != "
            f"config train_size {config['train_size']}"
        )
    if len(val_ids) != config["checkpoint_selection"]["eval_subset_size"]:
        raise RuntimeError(
            f"val sample ids count {len(val_ids)} != "
            "config eval_subset_size "
            f"{config['checkpoint_selection']['eval_subset_size']}"
        )
    if len(set(train_ids)) != len(train_ids):
        raise RuntimeError("train sample ids contain duplicates")
    if len(set(val_ids)) != len(val_ids):
        raise RuntimeError("val sample ids contain duplicates")

    train_rows = _load_csv_rows(splits_dir / "train.csv")
    val_rows = _load_csv_rows(splits_dir / "val.csv")

    # 5/6: every sample id must belong to its own split.
    missing_train = [id_ for id_ in train_ids if id_ not in train_rows]
    if missing_train:
        raise RuntimeError(
            f"{len(missing_train)} train sample ids not found in "
            "train.csv"
        )
    missing_val = [id_ for id_ in val_ids if id_ not in val_rows]
    if missing_val:
        raise RuntimeError(
            f"{len(missing_val)} val sample ids not found in val.csv"
        )

    # 7: the two frozen subsets must not overlap.
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise RuntimeError(
            f"train/val sample ids overlap: {len(overlap)} ids"
        )

    # 8: label distributions must match the manifests.
    train_dist = _label_distribution(train_ids, train_rows)
    val_dist = _label_distribution(val_ids, val_rows)
    if train_dist != train_manifest["label_distribution"]:
        raise RuntimeError(
            "train label distribution mismatch: "
            f"manifest={train_manifest['label_distribution']} "
            f"actual={train_dist}"
        )
    if val_dist != val_manifest["label_distribution"]:
        raise RuntimeError(
            "val label distribution mismatch: "
            f"manifest={val_manifest['label_distribution']} "
            f"actual={val_dist}"
        )

    # 9: test.csv / eval_test.csv sample content is never read here.
    return {
        "train_rows": len(train_ids),
        "train_sample_ids_sha256": train_ids_sha256,
        "train_csv_sha256": split_manifest["splits"]["train"]["csv_sha256"],
        "train_label_distribution": train_dist,
        "val_rows": len(val_ids),
        "val_sample_ids_sha256": val_ids_sha256,
        "val_csv_sha256": split_manifest["splits"]["val"]["csv_sha256"],
        "val_label_distribution": val_dist,
        "overlap": len(overlap),
        "test_accessed_for_sample_content": False,
        "eval_test_accessed_for_sample_content": False,
    }


def load_adapter_runtime(adapter_dir: Path):
    """Load BF16 base + QLoRA adapter with the frozen eval dtype."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the QLoRA eval")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # Frozen decoding is greedy; sampling defaults are unset exactly
    # like the Base eval so they cannot leak into generation.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return tokenizer, model


def evaluate_samples(tokenizer, model, samples) -> list[dict]:
    """Run the frozen per-sample generation protocol over samples.

    samples is a list of (sample_id, gold_label, title, summary).
    """
    results = []
    for sample_id, gold_label, title, summary in samples:
        raw_output, latency_seconds = generate_one(
            tokenizer, model, title, summary
        )
        prediction, valid_output = parse_final_label(raw_output)
        results.append(
            {
                "sample_id": sample_id,
                "gold_label": gold_label,
                "prediction": prediction,
                "valid_output": valid_output,
                "raw_output": raw_output,
                "latency_seconds": latency_seconds,
            }
        )
    return results


def summarize_results(results: list[dict]) -> dict:
    """Compute the frozen metrics over per-sample eval results.

    Invalid predictions (-1) stay in the denominator exactly like the
    Base eval.
    """
    gold_labels = [result["gold_label"] for result in results]
    predictions = [result["prediction"] for result in results]
    valid_flags = [result["valid_output"] for result in results]
    latencies = [result["latency_seconds"] for result in results]

    invalid_count = sum(1 for valid in valid_flags if not valid)
    valid_output_rate = (len(results) - invalid_count) / len(results)
    sorted_latencies = sorted(latencies)

    return {
        "rows": len(results),
        "macro_f1": macro_f1(gold_labels, predictions),
        "accuracy": accuracy(gold_labels, predictions),
        "valid_output_rate": valid_output_rate,
        "invalid_count": invalid_count,
        "total_inference_seconds": float(sum(latencies)),
        "mean_latency_ms_per_sample": (
            float(sum(latencies)) / len(results) * 1000
        ),
        "p50_latency_ms": statistics.median(sorted_latencies) * 1000,
        "p95_latency_ms": sorted_latencies[
            int(0.95 * (len(sorted_latencies) - 1))
        ]
        * 1000,
        "gold_label_distribution": {
            str(label): gold_labels.count(label)
            for label in RISK_LABELS
        },
        "prediction_label_distribution": {
            **{
                str(label): predictions.count(label)
                for label in RISK_LABELS
            },
            "invalid": invalid_count,
        },
    }


def select_checkpoint(checkpoint_results: list[dict]) -> dict:
    """Frozen 4B selection rule.

    Highest Val Macro-F1 wins; an exact tie breaks toward the smaller
    global_step. Only this rule may pick the formal adapter.
    """
    if not checkpoint_results:
        raise RuntimeError("no checkpoint results to select from")
    return max(
        checkpoint_results,
        key=lambda result: (
            result["val_macro_f1"],
            -result["global_step"],
        ),
    )


def build_confusion_matrix(gold_labels, predictions, valid_flags):
    """Fixed five-class matrix over valid predictions only."""
    from sklearn.metrics import confusion_matrix

    valid_gold = [
        label
        for label, valid in zip(gold_labels, valid_flags)
        if valid
    ]
    valid_pred = [
        label
        for label, valid in zip(predictions, valid_flags)
        if valid
    ]
    return confusion_matrix(valid_gold, valid_pred, labels=RISK_LABELS)


def freeze_selected_adapter(
    adapter_dir: Path,
    checkpoint_result: dict,
    selection_file: Path,
    manifest_path: Path,
    config: dict | None = None,
) -> dict:
    """Freeze the selected adapter: manifest with file SHA256s."""
    adapter_safetensors = adapter_dir / "adapter_model.safetensors"
    adapter_config_json = adapter_dir / "adapter_config.json"
    if not adapter_safetensors.exists():
        raise RuntimeError(
            f"adapter_model.safetensors not found in {adapter_dir}"
        )
    if not adapter_config_json.exists():
        raise RuntimeError(
            f"adapter_config.json not found in {adapter_dir}"
        )

    config = config or {}
    train_ids_sha256 = config.get(
        "train_sample_ids_sha256", FROZEN_TRAIN_IDS_SHA256
    )

    manifest = {
        "purpose": "frozen formal Risk QLoRA adapter (Checkpoint 4B); "
        "eval_test may be read only after this manifest exists and "
        "verifies",
        "selected_checkpoint_path": str(adapter_dir),
        "selected_epoch": checkpoint_result["epoch"],
        "selected_global_step": checkpoint_result["global_step"],
        "adapter_model_safetensors": {
            "path": str(adapter_safetensors),
            "sha256": sha256_file(adapter_safetensors),
        },
        "adapter_config_json": {
            "path": str(adapter_config_json),
            "sha256": sha256_file(adapter_config_json),
        },
        "base_model_path": str(MODEL_PATH),
        "train_sample_ids_sha256": train_ids_sha256,
        "val_sample_ids_sha256": FROZEN_VAL_IDS_SHA256,
        "selection_metric": "val_macro_f1",
        "selection_rule": (
            "post_training_generative_val; highest Val1000 Macro-F1 "
            "(labels [1,2,3,4,5], average=macro, zero_division=0); "
            "exact tie -> smaller_global_step"
        ),
        "selection_result": {
            "val_macro_f1": checkpoint_result["val_macro_f1"],
            "accuracy": checkpoint_result["accuracy"],
            "valid_output_rate": checkpoint_result["valid_output_rate"],
            "invalid_count": checkpoint_result["invalid_count"],
        },
        "selection_file": str(selection_file),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def assert_adapter_frozen(manifest_path: Path) -> dict:
    """Gate eval_test access behind the frozen adapter manifest.

    Refuses unless the manifest exists and both adapter files match
    their recorded SHA256.
    """
    if not manifest_path.exists():
        raise RuntimeError(
            "selected_adapter_manifest.json not found; eval_test may "
            "not be read before the frozen Val1000 selection wrote "
            f"{manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("adapter_model_safetensors", "adapter_config_json"):
        entry = manifest[key]
        path = Path(entry["path"])
        if not path.exists():
            raise RuntimeError(
                f"{key} not found at {path}; the frozen adapter must "
                "not be moved or deleted"
            )
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"{key} SHA256 mismatch: "
                f"manifest={entry['sha256']} actual={actual}"
            )
    return manifest


def load_val_selection_samples(config: dict) -> list[tuple]:
    """Load the frozen Val1000 rows in the frozen ids-file order."""
    val_manifest_path = _resolve(
        REPO_ROOT, config["checkpoint_selection"]["eval_subset_manifest"]
    )
    val_manifest = json.loads(
        val_manifest_path.read_text(encoding="utf-8")
    )
    val_ids = _load_sample_ids(
        _resolve(REPO_ROOT, val_manifest["sample_ids_file"])
    )
    val_rows = _load_csv_rows(SPLITS_DIR / "val.csv")
    return [
        (
            sample_id,
            val_rows[sample_id]["label"],
            val_rows[sample_id]["title"],
            val_rows[sample_id]["summary"],
        )
        for sample_id in val_ids
    ]


def _discover_checkpoints(artifacts_dir: Path) -> list[Path]:
    checkpoints = []
    for path in artifacts_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        step_text = path.name[len("checkpoint-"):]
        if not step_text.isdigit():
            continue
        checkpoints.append(path)
    return sorted(
        checkpoints, key=lambda path: int(path.name.split("-")[1])
    )


def run_checkpoint_selection(
    artifacts_dir: Path = ARTIFACTS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """Eval the three epoch checkpoints on Val1000 and freeze the best."""
    config = load_frozen_config()
    data_summary = verify_frozen_training_data(config)

    checkpoints = _discover_checkpoints(artifacts_dir)
    if len(checkpoints) != 3:
        raise RuntimeError(
            "expected exactly 3 epoch checkpoints in "
            f"{artifacts_dir}, found {len(checkpoints)}"
        )

    val_samples = load_val_selection_samples(config)
    if len(val_samples) != config["checkpoint_selection"]["eval_subset_size"]:
        raise RuntimeError(
            f"Val selection rows {len(val_samples)} != "
            f"{config['checkpoint_selection']['eval_subset_size']}"
        )

    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_results = []
    for checkpoint_dir in checkpoints:
        trainer_state = json.loads(
            (checkpoint_dir / "trainer_state.json").read_text(
                encoding="utf-8"
            )
        )
        epoch = round(trainer_state["epoch"], 6)
        global_step = int(trainer_state["global_step"])

        print(
            f"[val-selection] evaluating {checkpoint_dir.name} "
            f"(epoch={epoch}, global_step={global_step}) ..."
        )
        tokenizer, model = load_adapter_runtime(checkpoint_dir)
        results = evaluate_samples(tokenizer, model, val_samples)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        summary = summarize_results(results)
        print(
            f"[val-selection] {checkpoint_dir.name}: "
            f"Macro-F1={summary['macro_f1']:.6f} "
            f"Acc={summary['accuracy']:.6f} "
            f"valid={summary['valid_output_rate']:.6f} "
            f"invalid={summary['invalid_count']}"
        )

        checkpoint_results.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "checkpoint_path": str(checkpoint_dir),
                "val_macro_f1": summary["macro_f1"],
                "accuracy": summary["accuracy"],
                "valid_output_rate": summary["valid_output_rate"],
                "invalid_count": summary["invalid_count"],
                "gold_label_distribution": summary[
                    "gold_label_distribution"
                ],
                "prediction_label_distribution": summary[
                    "prediction_label_distribution"
                ],
                "per_sample": [
                    {
                        "sample_id": result["sample_id"],
                        "gold_label": result["gold_label"],
                        "prediction": result["prediction"],
                        "valid_output": result["valid_output"],
                        "raw_output": result["raw_output"],
                    }
                    for result in results
                ],
            }
        )

    selected = select_checkpoint(checkpoint_results)

    selection = {
        "purpose": (
            "Val1000 generative checkpoint selection (Checkpoint 4B); "
            "the selected checkpoint is frozen for the formal eval"
        ),
        "selection_rule": (
            "post_training_generative_val: highest Val1000 Macro-F1 "
            "(labels [1,2,3,4,5], average=macro, zero_division=0); "
            "exact tie -> smaller global_step"
        ),
        "tie_break": "smaller_global_step",
        "metric": "val_macro_f1",
        "val_subset_rows": len(val_samples),
        "val_sample_ids_sha256": data_summary["val_sample_ids_sha256"],
        "checkpoints": checkpoint_results,
        "selected": {
            "epoch": selected["epoch"],
            "global_step": selected["global_step"],
            "checkpoint_path": selected["checkpoint_path"],
            "val_macro_f1": selected["val_macro_f1"],
            "accuracy": selected["accuracy"],
            "valid_output_rate": selected["valid_output_rate"],
            "invalid_count": selected["invalid_count"],
        },
    }
    (results_dir / "val_checkpoint_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = freeze_selected_adapter(
        adapter_dir=Path(selected["checkpoint_path"]),
        checkpoint_result=selected,
        selection_file=results_dir / "val_checkpoint_selection.json",
        manifest_path=results_dir / "selected_adapter_manifest.json",
        config=config,
    )

    print(
        f"[val-selection] selected: epoch={selected['epoch']} "
        f"global_step={selected['global_step']} "
        f"Val Macro-F1={selected['val_macro_f1']:.6f} "
        f"(rule: highest val_macro_f1, tie -> smaller global_step)"
    )
    print(
        f"[val-selection] adapter frozen to "
        f"{results_dir / 'selected_adapter_manifest.json'}"
    )
    return selection


def _write_confusion_matrix_png(
    gold_labels,
    predictions,
    valid_flags,
    path: Path,
    title_prefix: str,
) -> None:
    """Render the five-class matrix; only valid predictions enter it.

    The title reports the formal denominator of 500 plus the valid and
    invalid counts so the matrix never appears to cover all rows while
    silently dropping invalid outputs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    blue_ramp = {
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

    valid_n = sum(1 for valid in valid_flags if valid)
    invalid_n = len(gold_labels) - valid_n
    matrix = build_confusion_matrix(gold_labels, predictions, valid_flags)
    cmap = LinearSegmentedColormap.from_list(
        "blue-ramp",
        [blue_ramp[step] for step in sorted(blue_ramp)],
    )

    figure, axis = plt.subplots(figsize=(6.4, 5.6))
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
        f"{title_prefix} Confusion Matrix\n"
        f"(formal denominator=500, valid={valid_n}, "
        f"invalid={invalid_n})"
    )
    figure.colorbar(image, ax=axis, label="count")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_formal_eval(
    output_dir: Path,
    manifest_path: Path | None = None,
) -> dict:
    """Formal fixed-500-row eval; gated by the frozen adapter manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = assert_adapter_frozen(
        manifest_path or SELECTED_ADAPTER_MANIFEST
    )

    verify_frozen_eval_test()

    samples = load_samples(SPLITS_DIR / "eval_test.csv")
    if len(samples) != 500:
        raise RuntimeError(f"eval_test rows must be 500, got {len(samples)}")

    adapter_dir = Path(manifest["selected_checkpoint_path"])
    print(f"[formal] loading selected adapter from {adapter_dir} ...")
    tokenizer, model = load_adapter_runtime(adapter_dir)

    results = evaluate_samples(tokenizer, model, samples)
    summary = summarize_results(results)

    sample_ids = [result["sample_id"] for result in results]
    gold_labels = [result["gold_label"] for result in results]
    predictions = [result["prediction"] for result in results]
    valid_flags = [result["valid_output"] for result in results]
    raw_outputs = [result["raw_output"] for result in results]
    latencies = [result["latency_seconds"] for result in results]

    with (output_dir / "risk_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "sample_id",
                "gold_label",
                "prediction",
                "valid_output",
                "latency_seconds",
            ]
        )
        for sample_id, gold_label, prediction, valid_output, latency in zip(
            sample_ids,
            gold_labels,
            predictions,
            valid_flags,
            latencies,
        ):
            writer.writerow(
                [
                    sample_id,
                    gold_label,
                    prediction,
                    valid_output,
                    repr(latency),
                ]
            )

    with (output_dir / "risk_raw_outputs.jsonl").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        for sample_id, raw_output, prediction, valid_output in zip(
            sample_ids,
            raw_outputs,
            predictions,
            valid_flags,
        ):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "raw_output": raw_output,
                        "parsed_label": prediction,
                        "valid_output": valid_output,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {
        "task": "risk",
        "model_name": "Qwen3-8B + QLoRA",
        "model_identifier_or_path": str(MODEL_PATH),
        "model_revision": None,
        "adapter_path": str(adapter_dir),
        "adapter_model_safetensors_sha256": manifest[
            "adapter_model_safetensors"
        ]["sha256"],
        "adapter_config_json_sha256": manifest["adapter_config_json"][
            "sha256"
        ],
        "selected_epoch": manifest["selected_epoch"],
        "selected_global_step": manifest["selected_global_step"],
        "val_macro_f1": manifest["selection_result"]["val_macro_f1"],
        "eval_rows": len(samples),
        "eval_test_sha256": FROZEN_EVAL_TEST_SHA256,
        "macro_f1": summary["macro_f1"],
        "accuracy": summary["accuracy"],
        "valid_output_rate": summary["valid_output_rate"],
        "invalid_count": summary["invalid_count"],
        "total_inference_seconds": summary["total_inference_seconds"],
        "mean_latency_ms_per_sample": summary[
            "mean_latency_ms_per_sample"
        ],
        "p50_latency_ms": summary["p50_latency_ms"],
        "p95_latency_ms": summary["p95_latency_ms"],
        "gold_label_distribution": summary["gold_label_distribution"],
        "prediction_label_distribution": summary[
            "prediction_label_distribution"
        ],
        "prompt_version": RISK_PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "enable_thinking": False,
        "do_sample": False,
        "temperature": "not_used",
        "top_p": "not_used",
        "top_k": "not_used",
        "max_input_length": 1024,
        "max_new_tokens": 32,
        "dtype": "bfloat16",
        "quantization": "none",
        "inference_mode": "BF16_BASE_PLUS_ADAPTER",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(model.device),
        "gpu_name": torch.cuda.get_device_name(0),
        "git_commit": git_commit_sha(REPO_ROOT),
        "working_tree_dirty": git_working_tree_dirty(REPO_ROOT),
    }
    (output_dir / "risk_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_confusion_matrix_png(
        gold_labels,
        predictions,
        valid_flags,
        output_dir / "risk_confusion_matrix.png",
        title_prefix="Qwen3-8B + QLoRA",
    )

    print(f"Macro-F1: {summary['macro_f1']:.6f}")
    print(f"Accuracy: {summary['accuracy']:.6f}")
    print(f"valid_output_rate: {summary['valid_output_rate']:.6f}")
    print(f"invalid_count: {summary['invalid_count']}")
    print(
        "prediction distribution: "
        + json.dumps(summary["prediction_label_distribution"])
    )
    print(f"outputs written to: {output_dir}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.qwen_qlora",
        description=(
            "Formal Risk QLoRA checkpoint selection and evaluation "
            "(Checkpoint 4B)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    selection = subparsers.add_parser(
        "select-checkpoint",
        help=(
            "Eval the three epoch checkpoints on Val1000 and freeze "
            "the best adapter manifest"
        ),
    )
    selection.add_argument(
        "--artifacts-dir",
        default=str(ARTIFACTS_DIR),
        help="Formal training output directory",
    )
    selection.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Directory that receives the selection and manifest files",
    )

    formal = subparsers.add_parser(
        "formal-eval",
        help="Formal fixed-500-row eval_test evaluation",
    )
    formal.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Directory that receives the formal evaluation artifacts",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select-checkpoint":
        run_checkpoint_selection(
            artifacts_dir=Path(args.artifacts_dir),
            results_dir=Path(args.results_dir),
        )
    elif args.command == "formal-eval":
        run_formal_eval(output_dir=Path(args.output_dir))
    else:  # pragma: no cover - argparse prevents this
        raise RuntimeError(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
