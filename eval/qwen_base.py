"""Qwen3-8B Base evaluation over the frozen Risk eval_test.

Frozen inference protocol (never tuned against eval_test):

- checkpoint: the same local Qwen3-8B base the QLoRA path records in
  qwen_risk_model/adapter_config.json (base_model_name_or_path)
- prompt: eval.prompt version 1 (title + summary -> FINAL_LABEL=X)
- parser: eval.parser version 1 (exactly one FINAL_LABEL=[1-5],
  anything else invalid, invalid counts as a wrong answer)
- enable_thinking=False, do_sample=False, max_new_tokens=32
- temperature/top_p/top_k: not_used (do_sample=False makes them moot)
- dtype bfloat16, no quantization
- latency: time.perf_counter around model.generate with
  torch.cuda.synchronize before and after; model/tokenizer/CSV
  loading is excluded
"""

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.metrics import RISK_LABELS, accuracy, macro_f1
from eval.parser import PARSER_VERSION, parse_final_label
from eval.prompt import (
    RISK_PROMPT_VERSION,
    apply_chat_template,
    build_messages,
)
from preprocess.make_splits import (
    git_commit_sha,
    git_working_tree_dirty,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits" / "risk"
MODEL_PATH = REPO_ROOT / "Qwen"

FROZEN_EVAL_TEST_SHA256 = (
    "b052920047da97cc3e7e7ab4df382e79ebc9ab49e4fd466e7364807972272986"
)

MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 32

FROZEN_PROTOCOL = {
    "prompt_version": RISK_PROMPT_VERSION,
    "parser_version": PARSER_VERSION,
    "enable_thinking": False,
    "do_sample": False,
    "temperature": "not_used",
    "top_p": "not_used",
    "top_k": "not_used",
    "max_input_length": MAX_INPUT_LENGTH,
    "max_new_tokens": MAX_NEW_TOKENS,
    "dtype": "bfloat16",
    "quantization": "none",
}


def verify_frozen_eval_test() -> dict:
    """Fail loudly unless eval_test matches the frozen checkpoint."""
    manifest_path = SPLITS_DIR / "split_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

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
    return manifest


def load_samples(csv_path: Path, limit: int | None = None):
    """Load (sample_id, gold_label, title, summary) in file order."""
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    row["sample_id"],
                    int(row["label"]),
                    row["title"],
                    row["summary"],
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_runtime() -> tuple:
    """Load tokenizer and model with the frozen dtype."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3-8B eval")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Frozen decoding is greedy (do_sample=False); the packaged
    # sampling defaults are explicitly unset so they cannot leak into
    # any generation path. temperature/top_p/top_k stay "not_used".
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return tokenizer, model


def generate_one(
    tokenizer,
    model,
    title: str,
    summary: str,
) -> tuple[str, float]:
    """Generate one sample's raw output; returns (text, latency_seconds)."""
    messages = build_messages(title, summary)
    prompt_text = apply_chat_template(tokenizer, messages)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=pad_token_id,
        )
        torch.cuda.synchronize()
        latency_seconds = time.perf_counter() - start

    generated_ids = outputs[0][input_ids.shape[1]:]
    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )
    return raw_output, latency_seconds


def run_smoke(output_dir: Path, sample_count: int) -> None:
    """Smoke test on a few Train rows; never touches eval_test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_frozen_eval_test()

    samples = load_samples(SPLITS_DIR / "train.csv", limit=sample_count)
    tokenizer, model = load_runtime()

    results = []
    for index, (sample_id, gold_label, title, summary) in enumerate(
        samples, start=1
    ):
        raw_output, latency_seconds = generate_one(
            tokenizer, model, title, summary
        )
        prediction, valid_output = parse_final_label(raw_output)
        results.append(
            {
                "smoke_index": index,
                "sample_id": sample_id,
                "gold_label": gold_label,
                "raw_output": raw_output,
                "parsed_label": prediction,
                "valid_output": valid_output,
                "latency_seconds": latency_seconds,
            }
        )
        print(
            f"[smoke {index}/{len(samples)}] "
            f"gold={gold_label} parsed={prediction} "
            f"valid={valid_output} latency={latency_seconds:.4f}s "
            f"raw={raw_output!r}"
        )

    with (output_dir / "smoke_outputs.jsonl").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        for result in results:
            handle.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    print(f"smoke outputs written to: {output_dir}")
    print("used eval_test: False")
    print(
        "smoke valid rate: "
        f"{sum(1 for r in results if r['valid_output'])}"
        f"/{len(results)}"
    )


def run_formal(output_dir: Path) -> None:
    """Run the single formal 500-row eval and write all artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_frozen_eval_test()

    samples = load_samples(SPLITS_DIR / "eval_test.csv")
    if len(samples) != 500:
        raise RuntimeError(
            f"eval_test rows must be 500, got {len(samples)}"
        )

    tokenizer, model = load_runtime()

    sample_ids = []
    gold_labels = []
    predictions = []
    valid_flags = []
    latencies = []
    raw_outputs = []

    for sample_id, gold_label, title, summary in samples:
        raw_output, latency_seconds = generate_one(
            tokenizer, model, title, summary
        )
        prediction, valid_output = parse_final_label(raw_output)

        sample_ids.append(sample_id)
        gold_labels.append(gold_label)
        predictions.append(prediction)
        valid_flags.append(valid_output)
        latencies.append(latency_seconds)
        raw_outputs.append(raw_output)

    total_inference_seconds = float(sum(latencies))
    mean_latency_ms_per_sample = (
        total_inference_seconds / len(samples) * 1000
    )
    invalid_count = sum(1 for valid in valid_flags if not valid)
    valid_output_rate = (
        len(samples) - invalid_count
    ) / len(samples)

    eval_macro_f1 = macro_f1(gold_labels, predictions)
    eval_accuracy = accuracy(gold_labels, predictions)

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

    sorted_latencies = sorted(latencies)
    metrics = {
        "task": "risk",
        "model_name": "Qwen3-8B Base",
        "model_identifier_or_path": str(MODEL_PATH),
        "model_revision": None,
        "eval_rows": len(samples),
        "eval_test_sha256": FROZEN_EVAL_TEST_SHA256,
        "macro_f1": eval_macro_f1,
        "accuracy": eval_accuracy,
        "valid_output_rate": valid_output_rate,
        "invalid_count": invalid_count,
        "total_inference_seconds": total_inference_seconds,
        "mean_latency_ms_per_sample": mean_latency_ms_per_sample,
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
        "prompt_version": RISK_PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "enable_thinking": False,
        "do_sample": False,
        "temperature": "not_used",
        "top_p": "not_used",
        "top_k": "not_used",
        "max_input_length": MAX_INPUT_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "dtype": "bfloat16",
        "quantization": "none",
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
    )

    print(f"Macro-F1: {eval_macro_f1:.6f}")
    print(f"Accuracy: {eval_accuracy:.6f}")
    print(f"valid_output_rate: {valid_output_rate:.6f}")
    print(f"invalid_count: {invalid_count}")
    print(
        f"total_inference_seconds: {total_inference_seconds:.6f}"
    )
    print(
        "mean_latency_ms_per_sample: "
        f"{mean_latency_ms_per_sample:.6f}"
    )
    print(
        "prediction distribution: "
        + json.dumps(metrics["prediction_label_distribution"])
    )
    print(f"outputs written to: {output_dir}")


def _write_confusion_matrix_png(
    gold_labels,
    predictions,
    valid_flags,
    path: Path,
) -> None:
    """Render the confusion matrix, honest about invalid outputs.

    Only valid five-class predictions enter the matrix cells; the
    title reports the formal denominator of 500 plus the valid and
    invalid counts so the matrix never appears to cover all 500 rows
    while silently dropping invalid ones.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import confusion_matrix

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

    valid_mask = [
        valid for valid in valid_flags
    ]
    valid_gold = [
        label
        for label, valid in zip(gold_labels, valid_mask)
        if valid
    ]
    valid_pred = [
        label
        for label, valid in zip(predictions, valid_mask)
        if valid
    ]
    valid_n = len(valid_gold)
    invalid_n = len(gold_labels) - valid_n

    matrix = confusion_matrix(
        valid_gold, valid_pred, labels=RISK_LABELS
    )
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
        "Qwen3-8B Base Confusion Matrix\n"
        f"(formal denominator=500, valid={valid_n}, "
        f"invalid={invalid_n})"
    )
    figure.colorbar(image, ax=axis, label="count")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.qwen_base",
        description=(
            "Qwen3-8B Base evaluation over the frozen Risk eval_test."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that receives the evaluation artifacts",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=None,
        help=(
            "Run a smoke test on this many Train rows instead of the "
            "formal eval; eval_test is never used in smoke mode"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke is not None:
        run_smoke(Path(args.output_dir), args.smoke)
    else:
        run_formal(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
