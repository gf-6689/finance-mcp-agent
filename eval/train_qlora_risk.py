"""Formal Risk QLoRA training (Checkpoint 4B).

Frozen protocol (config/risk_qlora_training.json):

- data: the frozen Train20k sample ids, verified against the split
  manifest and the 4A SHA256 constants before any training starts;
- sequences: build_training_sequences + TargetOnlyDataCollator from
  the frozen eval/qlora_smoke module (Prompt v1, target-only loss,
  prompt truncation only);
- 4-bit QLoRA: nf4 + double quant + bf16 compute, LoRA r=16 /
  alpha=32 / dropout=0.1 / bias=none over the 7 attention+MLP
  projections;
- Trainer: save_strategy=epoch, eval_strategy=no,
  load_best_model_at_end=false, metric_for_best_model=null,
  greater_is_better=null, save_total_limit=3 — no internal best-model
  selection; the formal checkpoint is chosen only by the
  post-training generative Val1000 Macro-F1 (eval/qwen_qlora.py);
- 20,000 rows x 3 epochs; checkpoints at epoch ends are the three
  candidates and all must exist at the end.

This module never touches val/test/eval_test sample content.
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from eval.qlora_smoke import (
    SMOKE_TARGET_MODULES,
    TargetOnlyDataCollator,
    build_training_sequences,
)
from eval.qwen_qlora import (
    ARTIFACTS_DIR,
    REPO_ROOT,
    SPLITS_DIR,
    TRAINING_DIR,
    load_frozen_config,
    verify_frozen_training_data,
)
from preprocess.make_splits import git_commit_sha, git_working_tree_dirty

# Frozen protocol section 5; not a config key, pinned here.
DATA_SEED = 42
LOGGING_STEPS = 25


def build_training_arguments(
    config: dict,
    output_dir: Path,
) -> TrainingArguments:
    """Map the frozen config to TrainingArguments.

    The Trainer internal best-model selection is fully disabled:
    save every epoch, never evaluate, never load best at end, and keep
    all three epoch checkpoints.
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        optim=config["optimizer"],
        lr_scheduler_type=config["scheduler"],
        warmup_ratio=config["warmup_ratio"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=LOGGING_STEPS,
        save_strategy=config["save_strategy"],
        eval_strategy=config["eval_strategy"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        save_total_limit=config["save_total_limit"],
        report_to=[],
        remove_unused_columns=False,
        seed=config["seed"],
        data_seed=DATA_SEED,
        dataloader_pin_memory=False,
    )


def load_training_rows(
    config: dict,
    training_dir: Path = TRAINING_DIR,
    splits_dir: Path = SPLITS_DIR,
) -> list[dict]:
    """Load the frozen Train20k rows in the frozen ids-file order."""
    import csv as csv_module

    train_manifest = json.loads(
        (training_dir / "risk_qlora_train_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    train_ids = (
        (training_dir / "risk_qlora_train_sample_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if len(train_ids) != config["train_size"]:
        raise RuntimeError(
            f"train ids {len(train_ids)} != {config['train_size']}"
        )

    rows_by_id = {}
    with (splits_dir / "train.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv_module.DictReader(handle):
            rows_by_id[row["sample_id"]] = row

    missing = [id_ for id_ in train_ids if id_ not in rows_by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} train sample ids not found in train.csv"
        )

    return [
        {
            "sample_id": sample_id,
            "title": rows_by_id[sample_id]["title"],
            "summary": rows_by_id[sample_id]["summary"],
            "label": int(rows_by_id[sample_id]["label"]),
        }
        for sample_id in train_ids
    ]


class EpochProgressCallback(TrainerCallback):
    """Record wall-clock elapsed time per epoch."""

    def __init__(self):
        self.epoch_events = []
        self._epoch_begin = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_begin = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_events.append(
            {
                "epoch": int(round(state.epoch)),
                "global_step": int(state.global_step),
                "elapsed_seconds": time.perf_counter() - self._epoch_begin,
            }
        )


def _per_epoch_stats(trainer, epoch_events, effective_batch_size):
    """Derive per-epoch loss/lr/step from the training log history."""
    stats = []
    losses = [
        entry
        for entry in trainer.state.log_history
        if "loss" in entry and "epoch" in entry
    ]
    epoch_count = max(
        (event["epoch"] for event in epoch_events), default=0
    )
    for epoch in range(1, epoch_count + 1):
        epoch_losses = [
            entry for entry in losses
            if math.ceil(max(entry["epoch"], 1e-9)) == epoch
        ]
        event = next(
            (
                item for item in epoch_events
                if item["epoch"] == epoch
            ),
            None,
        )
        steps = len(epoch_losses) * LOGGING_STEPS
        stats.append(
            {
                "epoch": epoch,
                "global_step": (
                    max(entry["step"] for entry in epoch_losses)
                    if epoch_losses
                    else None
                ),
                "mean_train_loss": (
                    sum(entry["loss"] for entry in epoch_losses)
                    / len(epoch_losses)
                    if epoch_losses
                    else None
                ),
                "learning_rate": (
                    epoch_losses[-1].get("learning_rate")
                    if epoch_losses
                    else None
                ),
                "elapsed_seconds": (
                    event["elapsed_seconds"] if event else None
                ),
                "examples_per_second": (
                    steps * effective_batch_size / event["elapsed_seconds"]
                    if event and event["elapsed_seconds"]
                    else None
                ),
                "nan_or_inf": any(
                    not math.isfinite(entry["loss"])
                    for entry in epoch_losses
                ),
            }
        )
    return stats


def run_training(
    config: dict,
    output_dir: Path,
    resume_from_checkpoint: Path | None = None,
) -> dict:
    """Run the single formal training; never touches eval_test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # The nine frozen-data checks run before any training work.
    data_summary = verify_frozen_training_data(config)
    print("frozen training data verified:")
    print(json.dumps(data_summary, indent=2, ensure_ascii=False))

    rows = load_training_rows(config)
    if len(rows) != config["train_size"]:
        raise RuntimeError(
            f"training rows {len(rows)} != {config['train_size']}"
        )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    sequences = build_training_sequences(
        rows, tokenizer, config["max_seq_length"]
    )
    token_lengths = [len(seq["input_ids"]) for seq in sequences]
    token_length_stats = {
        "min": min(token_lengths),
        "p50": sorted(token_lengths)[len(token_lengths) // 2],
        "p90": sorted(token_lengths)[int(0.90 * (len(token_lengths) - 1))],
        "p99": sorted(token_lengths)[int(0.99 * (len(token_lengths) - 1))],
        "max": max(token_lengths),
        "mean": sum(token_lengths) / len(token_lengths),
        "truncated": sum(
            1 for length in token_lengths if length == config["max_seq_length"]
        ),
    }
    print(f"training sequence token lengths: {token_length_stats}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=SMOKE_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if config["gradient_checkpointing"]:
        model.config.use_cache = False

    training_args = build_training_arguments(config, output_dir)
    dataset = Dataset.from_list(sequences)
    epoch_progress = EpochProgressCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=TargetOnlyDataCollator(tokenizer),
        tokenizer=tokenizer,
        callbacks=[epoch_progress],
    )

    effective_batch_size = (
        config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        oom = False
    except torch.cuda.OutOfMemoryError:
        oom = True
        raise
    finally:
        elapsed_seconds = time.perf_counter() - start
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    trainer.save_model()
    print(f"final adapter saved to: {output_dir}")

    per_epoch = _per_epoch_stats(
        trainer, epoch_progress.epoch_events, effective_batch_size
    )
    for entry in per_epoch:
        print(
            f"[epoch {entry['epoch']}] global_step={entry['global_step']} "
            f"train_loss={entry['mean_train_loss']:.6f} "
            f"lr={entry['learning_rate']} "
            f"elapsed={entry['elapsed_seconds']:.1f}s "
            f"examples/s={entry['examples_per_second']:.3f} "
            f"nan_or_inf={entry['nan_or_inf']}"
        )

    checkpoints = []
    for path in sorted(
        (p for p in output_dir.iterdir() if p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[1]),
    ):
        adapter_ok = (path / "adapter_model.safetensors").exists() and (
            path / "adapter_config.json"
        ).exists()
        trainer_state = json.loads(
            (path / "trainer_state.json").read_text(encoding="utf-8")
        )
        checkpoints.append(
            {
                "path": str(path),
                "name": path.name,
                "global_step": int(trainer_state["global_step"]),
                "epoch": trainer_state["epoch"],
                "adapter_files_present": adapter_ok,
            }
        )
    print(f"checkpoint dirs: {[c['name'] for c in checkpoints]}")
    if len(checkpoints) != config["epochs"]:
        raise RuntimeError(
            f"expected {config['epochs']} epoch checkpoints, "
            f"found {len(checkpoints)}"
        )
    for checkpoint in checkpoints:
        if not checkpoint["adapter_files_present"]:
            raise RuntimeError(
                f"checkpoint {checkpoint['name']} missing adapter files"
            )

    nan_or_inf_any = any(entry["nan_or_inf"] for entry in per_epoch)
    total_examples = (
        config["train_size"] * config["epochs"]
    )
    summary = {
        "protocol_version": config["protocol_version"],
        "frozen_at_checkpoint": config["frozen_at_checkpoint"],
        "started_at_utc": started_at,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "base_model": config["base_model"],
        "train_size": config["train_size"],
        "epochs": config["epochs"],
        "effective_batch_size": effective_batch_size,
        "max_seq_length": config["max_seq_length"],
        "seed": config["seed"],
        "data_seed": DATA_SEED,
        "data_verification": data_summary,
        "token_length_stats": token_length_stats,
        "per_epoch": per_epoch,
        "checkpoints": checkpoints,
        "total_elapsed_seconds": elapsed_seconds,
        "examples_per_second": total_examples / elapsed_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "nan_or_inf_in_loss": nan_or_inf_any,
        "oom": oom,
        "resumed_from_checkpoint": (
            str(resume_from_checkpoint)
            if resume_from_checkpoint
            else None
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "git_commit": git_commit_sha(REPO_ROOT),
        "working_tree_dirty": git_working_tree_dirty(REPO_ROOT),
    }
    (output_dir / "training_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"total_elapsed_seconds: {elapsed_seconds:.1f}")
    print(
        f"examples_per_second: {total_examples / elapsed_seconds:.3f}"
    )
    print(f"peak_gpu_memory_mb: {peak_gpu_memory_mb:.0f}")
    print(f"nan_or_inf_in_loss: {nan_or_inf_any}")
    print(f"oom: {oom}")
    print(f"summary written to: {output_dir / 'training_run_summary.json'}")
    print("used eval_test: False")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.train_qlora_risk",
        description=(
            "Formal Risk QLoRA training (Checkpoint 4B, frozen config)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ARTIFACTS_DIR),
        help="Fresh directory for the formal run",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Resume from a checkpoint of THIS formal output dir only "
            "(engineering interruption recovery)"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    resume_path = None
    if args.resume_from_checkpoint is not None:
        resume_path = Path(args.resume_from_checkpoint)
        if not resume_path.is_relative_to(output_dir):
            raise SystemExit(
                "resume checkpoint must be inside the same formal "
                f"output dir ({output_dir}); got {resume_path}"
            )
    run_training(
        load_frozen_config(),
        output_dir=output_dir,
        resume_from_checkpoint=resume_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
