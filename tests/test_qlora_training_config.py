"""Config consistency tests for the formal QLoRA training (Checkpoint 4B).

Guards the frozen trainer/checkpoint-selection protocol: the Trainer
must never run its own internal best-model selection (no eval loop, no
load_best_model_at_end), and the formal checkpoint must be chosen only
by the post-training generative Val1000 Macro-F1.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "risk_qlora_training.json"

FROZEN_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_trainer_internal_best_model_selection_is_disabled():
    config = _load_config()

    assert config["eval_strategy"] == "no"
    assert config["load_best_model_at_end"] is False
    assert config["metric_for_best_model"] is None
    assert config["greater_is_better"] is None


def test_trainer_saves_every_epoch_and_keeps_all_three_checkpoints():
    config = _load_config()

    assert config["save_strategy"] == "epoch"
    assert config["save_total_limit"] >= 3


def test_checkpoint_selection_is_post_training_generative_val_macro_f1():
    config = _load_config()
    selection = config["checkpoint_selection"]

    assert config["checkpoint_selection_mode"] == (
        "post_training_generative_val"
    )
    assert config["checkpoint_selection_metric"] == "val_macro_f1"
    assert config["checkpoint_selection_tie_break"] == (
        "smaller_global_step"
    )
    assert selection["split"] == "val"
    assert selection["metric"].startswith("macro_f1")
    assert selection["tie_break"] == (
        "smaller global_step (earlier checkpoint)"
    )
    assert selection["eval_generation"] == {
        "enable_thinking": False,
        "do_sample": False,
        "max_new_tokens": 32,
        "parser_version": "1",
    }
    assert selection["test_used"] is False
    assert selection["eval_test_used"] is False


def test_frozen_training_hyperparameters_are_unchanged():
    config = _load_config()

    assert config["train_size"] == 20000
    assert config["epochs"] == 3
    assert config["learning_rate"] == 2e-5
    assert config["per_device_train_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 16
    assert config["effective_batch_size"] == 16
    assert config["max_seq_length"] == 1024
    assert config["seed"] == 42
    assert config["optimizer"] == "adamw_torch"
    assert config["scheduler"] == "linear"
    assert config["warmup_ratio"] == 0.03
    assert config["gradient_checkpointing"] is True
    assert config["bf16"] is True


def test_frozen_qlora_quantization_and_rank_are_unchanged():
    config = _load_config()

    assert config["load_in_4bit"] is True
    assert config["bnb_4bit_quant_type"] == "nf4"
    assert config["bnb_4bit_use_double_quant"] is True
    assert config["bnb_4bit_compute_dtype"] == "bfloat16"
    assert config["lora_r"] == 16
    assert config["lora_alpha"] == 32
    assert config["lora_dropout"] == 0.1
    assert config["target_modules"] == FROZEN_TARGET_MODULES


def test_frozen_training_data_references_are_unchanged():
    config = _load_config()

    assert config["train_sample_ids_sha256"] == (
        "e975eb686e33b7614076e8e233321592224df25e4686a51cf3650baa3b538a0b"
    )
    assert config["checkpoint_selection"]["eval_subset_manifest"] == (
        "data/training/risk_qlora_val_checkpoint_selection_manifest.json"
    )
    assert config["checkpoint_selection"]["eval_subset_size"] == 1000
