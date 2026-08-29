"""Minimal QLoRA engineering smoke (Checkpoint 4A).

Scope: prove the training -> save adapter -> reload -> inference
pipeline works, on a few Train rows only. Never touches eval_test,
never overwrites qwen_risk_model, never produces formal numbers.

- base: the same Qwen3-8B checkpoint as the frozen Base eval
- training: 4-bit QLoRA (nf4 + double quant + bf16 compute),
  LoRA r=16/alpha=32/dropout=0.1 over the 7 attention+MLP projections
- smoke scale: a few optimizer steps (default 16) on deterministic
  Train rows, adapter saved to tmp/qlora_smoke_adapter/
- reload: BF16 base + smoke adapter, inference on a few Val rows
  with Prompt v1 / parser v1 / enable_thinking=False / do_sample=False
- throughput: elapsed time over the smoke steps, effective batch =
  per_device_train_batch_size x gradient_accumulation_steps
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from eval.parser import parse_final_label
from eval.prompt import apply_chat_template, build_messages
from eval.qwen_base import generate_one

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits" / "risk"
MODEL_PATH = REPO_ROOT / "Qwen"

SMOKE_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class TargetOnlyDataCollator:
    """Pad causal-LM examples without replacing their loss masks."""

    def __init__(self, tokenizer):
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set")
        self.tokenizer = tokenizer

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("examples must not be empty")

        max_length = max(len(example["input_ids"]) for example in examples)
        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for example in examples:
            input_ids = list(example["input_ids"])
            labels = list(example["labels"])
            if len(input_ids) != len(labels):
                raise ValueError("input_ids and labels must have equal length")

            padding_length = max_length - len(input_ids)
            input_padding = [self.tokenizer.pad_token_id] * padding_length
            label_padding = [-100] * padding_length
            attention_padding = [0] * padding_length
            attention_tokens = [1] * len(input_ids)

            if self.tokenizer.padding_side == "left":
                padded_input_ids.append(input_padding + input_ids)
                padded_labels.append(label_padding + labels)
                attention_masks.append(attention_padding + attention_tokens)
            else:
                padded_input_ids.append(input_ids + input_padding)
                padded_labels.append(labels + label_padding)
                attention_masks.append(attention_tokens + attention_padding)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


def load_split_rows(csv_path: Path, limit: int) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def build_training_sequences(
    rows: list[dict],
    tokenizer,
    max_seq_length: int,
) -> list[dict]:
    """Build prompt+target sequences with labels masked to the target."""
    sequences = []
    for row in rows:
        messages = build_messages(row["title"], row["summary"])
        prompt_text = apply_chat_template(tokenizer, messages)
        target_text = (
            f"FINAL_LABEL={int(row['label'])}"
            + tokenizer.eos_token
        )
        prompt_ids = tokenizer.encode(
            prompt_text, add_special_tokens=False
        )
        target_ids = tokenizer.encode(
            target_text, add_special_tokens=False
        )
        if len(target_ids) > max_seq_length:
            raise ValueError(
                "max_seq_length cannot fit the complete target + EOS: "
                f"target={len(target_ids)} max={max_seq_length}"
            )

        prompt_budget = max_seq_length - len(target_ids)
        truncated_prompt_ids = prompt_ids[:prompt_budget]
        input_ids = truncated_prompt_ids + target_ids
        labels = [-100] * len(truncated_prompt_ids) + target_ids

        sequences.append(
            {"input_ids": input_ids, "labels": labels}
        )
    return sequences


def run_smoke(
    output_dir: Path,
    train_rows: int,
    val_rows: int,
    max_steps: int,
    grad_accum: int,
    max_seq_length: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_split_rows(SPLITS_DIR / "train.csv", train_rows)
    val_samples = load_split_rows(SPLITS_DIR / "val.csv", val_rows)
    print(f"train samples: {len(train_samples)}")
    print(f"val samples: {len(val_samples)}")
    print("eval_test accessed for sample content: False")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    sequences = build_training_sequences(
        train_samples, tokenizer, max_seq_length
    )
    token_lengths = [
        len(seq["input_ids"]) for seq in sequences
    ]
    print(
        "training sequence token lengths: "
        f"min={min(token_lengths)} max={max(token_lengths)} "
        f"mean={sum(token_lengths) / len(token_lengths):.1f}"
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=SMOKE_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=4,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=42,
        dataloader_pin_memory=False,
    )

    dataset = Dataset.from_list(sequences)
    data_collator = TargetOnlyDataCollator(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    elapsed_seconds = time.perf_counter() - start
    peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    trainer.save_model()
    print(f"adapter saved to: {output_dir}")

    del trainer
    del model
    torch.cuda.empty_cache()

    # --- reload: BF16 base + smoke adapter ---------------------------
    print("reloading BF16 base + smoke adapter ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    reloaded = PeftModel.from_pretrained(base_model, output_dir)
    reloaded.eval()
    reloaded.generation_config.temperature = None
    reloaded.generation_config.top_p = None
    reloaded.generation_config.top_k = None
    print(f"reloaded device: {reloaded.device}")
    print(
        "reloaded base dtype: "
        f"{next(reloaded.base_model.parameters()).dtype}"
    )

    inference_results = []
    for row in val_samples:
        raw_output, latency_seconds = generate_one(
            tokenizer, reloaded, row["title"], row["summary"]
        )
        prediction, valid_output = parse_final_label(raw_output)
        inference_results.append(
            {
                "sample_id": row["sample_id"],
                "gold_label": int(row["label"]),
                "raw_output": raw_output,
                "parsed_label": prediction,
                "valid_output": valid_output,
                "latency_seconds": latency_seconds,
            }
        )
        print(
            f"[infer] gold={row['label']} parsed={prediction} "
            f"valid={valid_output} raw={raw_output!r}"
        )

    training_examples = max_steps * grad_accum
    summary = {
        "base_model": str(MODEL_PATH),
        "training_examples": training_examples,
        "optimizer_steps": max_steps,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": training_examples / elapsed_seconds,
        "steps_per_second": max_steps / elapsed_seconds,
        "effective_batch_size": 1 * grad_accum,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "max_seq_length": max_seq_length,
        "adapter_dir": str(output_dir),
        "train_token_lengths": {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
        },
        "val_inference": inference_results,
        "overwrote_qwen_risk_model": False,
        "used_eval_test": False,
    }
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("--- smoke summary ---")
    print(f"training_examples: {training_examples}")
    print(f"optimizer_steps: {max_steps}")
    print(f"elapsed_seconds: {elapsed_seconds:.1f}")
    print(
        f"examples_per_second: "
        f"{training_examples / elapsed_seconds:.3f}"
    )
    print(f"steps_per_second: {max_steps / elapsed_seconds:.3f}")
    print(f"effective_batch_size: {1 * grad_accum}")
    print(f"peak_gpu_memory_mb: {peak_gpu_memory_mb:.0f}")
    print(
        "val inference valid rate: "
        f"{sum(1 for r in inference_results if r['valid_output'])}"
        f"/{len(inference_results)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.qlora_smoke",
        description="QLoRA engineering smoke (Checkpoint 4A).",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/qlora_smoke_adapter",
        help="Directory for the smoke adapter and summary",
    )
    parser.add_argument(
        "--train-rows", type=int, default=64,
        help="Deterministic Train rows used for the smoke",
    )
    parser.add_argument(
        "--val-rows", type=int, default=8,
        help="Deterministic Val rows used for reload inference",
    )
    parser.add_argument(
        "--max-steps", type=int, default=16,
        help="Optimizer steps for the smoke",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=512,
        help="Training sequence truncation length",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_smoke(
        Path(args.output_dir),
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        max_steps=args.max_steps,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
