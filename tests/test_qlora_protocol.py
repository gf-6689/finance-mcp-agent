"""Protocol tests for the formal QLoRA training/eval pipeline (Checkpoint 4B).

Covers the frozen-checkpoint-4B rules:

- frozen data verification (the nine pre-training checks, and never
  reading test/eval_test sample content);
- checkpoint selection by post-training generative Val Macro-F1 with
  the smaller-global-step tie-break;
- the selected-adapter manifest gate that must pass before eval_test
  may be read;
- the five-class confusion matrix that counts only valid predictions;
- the TrainingArguments mapping from the frozen config (Trainer
  internal best-model selection fully disabled).

No GPU is required: every test runs against tiny synthetic fixtures.
"""

import csv
import json
from pathlib import Path

import pytest

from eval import qwen_qlora
from eval.train_qlora_risk import build_training_arguments
from preprocess.make_splits import sha256_file, sha256_text


# ---------------------------------------------------------------------------
# synthetic frozen-world fixture
# ---------------------------------------------------------------------------

TRAIN_ROWS = [
    ("t1", 1),
    ("t2", 2),
    ("t3", 2),
    ("t4", 3),
    ("t5", 3),
    ("t6", 3),
]
VAL_ROWS = [
    ("v1", 1),
    ("v2", 2),
    ("v3", 3),
    ("v4", 3),
]


def _write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_id", "label", "title", "summary"])
        for sample_id, label in rows:
            writer.writerow([sample_id, label, "title", "summary"])


def _write_ids(path: Path, sample_ids: list[str]) -> None:
    path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")


def _full_distribution(label_counts: dict[int, int]) -> dict[str, int]:
    return {
        str(label): label_counts.get(label, 0)
        for label in (1, 2, 3, 4, 5)
    }


def _make_frozen_world(
    tmp_path: Path,
    train_ids=None,
    val_ids=None,
    val_rows=None,
) -> dict:
    """Build synthetic splits + training manifests; returns paths.

    test.csv and eval_test.csv are intentionally NOT created: the
    verifier must never touch them.
    """
    splits = tmp_path / "splits"
    splits.mkdir()
    training = tmp_path / "training"
    training.mkdir()

    val_rows = VAL_ROWS if val_rows is None else val_rows

    train_csv = splits / "train.csv"
    val_csv = splits / "val.csv"
    _write_csv(train_csv, TRAIN_ROWS)
    _write_csv(val_csv, val_rows)

    split_manifest = {
        "splits": {
            "train": {"csv_sha256": sha256_file(train_csv)},
            "val": {"csv_sha256": sha256_file(val_csv)},
        }
    }
    (splits / "split_manifest.json").write_text(
        json.dumps(split_manifest) + "\n", encoding="utf-8"
    )

    train_ids = ["t1", "t3", "t5"] if train_ids is None else train_ids
    val_ids = ["v1", "v2", "v3"] if val_ids is None else val_ids

    train_ids_path = training / "risk_qlora_train_sample_ids.txt"
    val_ids_path = training / "risk_qlora_val_checkpoint_selection_sample_ids.txt"
    _write_ids(train_ids_path, train_ids)
    _write_ids(val_ids_path, val_ids)

    train_manifest_path = training / "risk_qlora_train_manifest.json"
    val_manifest_path = training / (
        "risk_qlora_val_checkpoint_selection_manifest.json"
    )
    train_manifest = {
        "source": {"csv_sha256": sha256_file(train_csv)},
        "sample_ids_file": str(train_ids_path),
        "sample_ids_sha256": sha256_text("\n".join(train_ids)),
        "label_distribution": _label_distribution_of(
            [id_ for id_ in train_ids], TRAIN_ROWS
        ),
    }
    val_manifest = {
        "source": {"csv_sha256": sha256_file(val_csv)},
        "sample_ids_file": str(val_ids_path),
        "sample_ids_sha256": sha256_text("\n".join(val_ids)),
        "label_distribution": _label_distribution_of(
            [id_ for id_ in val_ids], val_rows
        ),
    }
    train_manifest_path.write_text(
        json.dumps(train_manifest, indent=2) + "\n", encoding="utf-8"
    )
    val_manifest_path.write_text(
        json.dumps(val_manifest, indent=2) + "\n", encoding="utf-8"
    )

    config = {
        "train_size": len(train_ids),
        "train_sample_ids_sha256": sha256_text("\n".join(train_ids)),
        "train_manifest": str(train_manifest_path),
        "checkpoint_selection": {
            "eval_subset_manifest": str(val_manifest_path),
            "eval_subset_size": len(val_ids),
        },
    }
    return {
        "splits": splits,
        "training": training,
        "config": config,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "train_ids_path": train_ids_path,
        "val_ids_path": val_ids_path,
    }


def _label_distribution_of(sample_ids, rows) -> dict[str, int]:
    labels = {sample_id: label for sample_id, label in rows}
    return _full_distribution(
        {
            label: sum(
                1 for id_ in sample_ids if labels.get(id_) == label
            )
            for label in (1, 2, 3, 4, 5)
        }
    )


# ---------------------------------------------------------------------------
# frozen data verification
# ---------------------------------------------------------------------------


def _verify(world, tmp_path, **kwargs):
    """Verify with the 4A constants disabled (synthetic fixtures)."""
    return qwen_qlora.verify_frozen_training_data(
        config=world["config"],
        repo_root=tmp_path,
        training_dir=world["training"],
        splits_dir=world["splits"],
        expected_train_ids_sha256=None,
        expected_val_ids_sha256=None,
        **kwargs,
    )


def test_verify_frozen_training_data_passes_and_never_touches_test_files(
    tmp_path,
):
    world = _make_frozen_world(tmp_path)

    summary = _verify(world, tmp_path)

    assert summary["train_rows"] == 3
    assert summary["val_rows"] == 3
    assert summary["overlap"] == 0
    assert summary["eval_test_accessed_for_sample_content"] is False
    assert summary["test_accessed_for_sample_content"] is False
    # test.csv / eval_test.csv were never created: reaching this point
    # proves the verifier never opened them.


def test_verify_rejects_train_csv_hash_mismatch(tmp_path):
    world = _make_frozen_world(tmp_path)
    with world["train_csv"].open("a", encoding="utf-8") as handle:
        handle.write("t7,2,title,summary\n")

    with pytest.raises(RuntimeError, match="train.csv SHA256 mismatch"):
        _verify(world, tmp_path)


def test_verify_rejects_val_csv_hash_mismatch(tmp_path):
    world = _make_frozen_world(tmp_path)
    with world["val_csv"].open("a", encoding="utf-8") as handle:
        handle.write("v9,2,title,summary\n")

    with pytest.raises(RuntimeError, match="val.csv SHA256 mismatch"):
        _verify(world, tmp_path)


def test_verify_rejects_train_sample_ids_file_tampering(tmp_path):
    world = _make_frozen_world(tmp_path)
    _write_ids(world["train_ids_path"], ["t2", "t3", "t5"])

    with pytest.raises(RuntimeError, match="sample_ids SHA256 mismatch"):
        _verify(world, tmp_path)


def test_verify_rejects_val_sample_ids_file_tampering(tmp_path):
    world = _make_frozen_world(tmp_path)
    _write_ids(world["val_ids_path"], ["v2", "v3", "v4"])

    with pytest.raises(RuntimeError, match="sample_ids SHA256 mismatch"):
        _verify(world, tmp_path)


def test_verify_rejects_id_missing_from_split(tmp_path):
    world = _make_frozen_world(tmp_path, train_ids=["t1", "t3", "t99"])

    with pytest.raises(RuntimeError, match="not found in train.csv"):
        _verify(world, tmp_path)


def test_verify_rejects_train_val_overlap(tmp_path):
    # A hypothetical split-regeneration bug puts the same id into both
    # CSVs; membership passes, so the overlap guard must fire.
    world = _make_frozen_world(
        tmp_path,
        train_ids=["t1", "t3", "t5"],
        val_ids=["t1", "v2", "v3"],
        val_rows=VAL_ROWS + [("t1", 2)],
    )

    with pytest.raises(RuntimeError, match="overlap"):
        _verify(world, tmp_path)


def test_verify_rejects_label_distribution_mismatch(tmp_path):
    world = _make_frozen_world(tmp_path)
    manifest_path = Path(world["config"]["train_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["label_distribution"]["2"] = 0
    manifest["label_distribution"]["3"] = 2
    manifest_path.write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="label distribution"):
        _verify(world, tmp_path)


# ---------------------------------------------------------------------------
# checkpoint selection rule (post-training generative Val Macro-F1)
# ---------------------------------------------------------------------------


def _checkpoint(global_step, macro_f1):
    return {
        "epoch": global_step // 1000,
        "global_step": global_step,
        "checkpoint_path": f"/fake/checkpoint-{global_step}",
        "val_macro_f1": macro_f1,
        "accuracy": 0.5,
        "valid_output_rate": 0.9,
        "invalid_count": 100,
    }


def test_select_checkpoint_prefers_higher_val_macro_f1():
    results = [
        _checkpoint(1250, 0.4000),
        _checkpoint(2500, 0.4213),
        _checkpoint(3750, 0.4150),
    ]

    selected = qwen_qlora.select_checkpoint(results)

    assert selected["global_step"] == 2500


def test_select_checkpoint_tie_breaks_to_smaller_global_step():
    results = [
        _checkpoint(1250, 0.4000),
        _checkpoint(2500, 0.4000),
        _checkpoint(3750, 0.3990),
    ]

    selected = qwen_qlora.select_checkpoint(results)

    assert selected["global_step"] == 1250


def test_select_checkpoint_rejects_empty_results():
    with pytest.raises(RuntimeError, match="no checkpoint"):
        qwen_qlora.select_checkpoint([])


# ---------------------------------------------------------------------------
# confusion matrix: valid predictions only, honest denominator
# ---------------------------------------------------------------------------


def test_confusion_matrix_sums_to_valid_count_only():
    gold = [3, 1, 5, 2, 4]
    pred = [3, 1, -1, 4, -1]
    valid = [True, True, False, True, False]

    matrix = qwen_qlora.build_confusion_matrix(gold, pred, valid)

    assert matrix.shape == (5, 5)
    assert int(matrix.sum()) == 3
    assert int(matrix.sum()) + (len(gold) - sum(valid)) == len(gold)
    assert int(matrix.sum()) != len(gold)


# ---------------------------------------------------------------------------
# selected-adapter manifest gate
# ---------------------------------------------------------------------------


def _make_adapter_dir(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(
        b"fake-adapter-weights\x00\x01"
    )
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "base_model_name_or_path": "/fake/Qwen",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_freeze_and_assert_adapter_manifest_roundtrip(tmp_path):
    adapter_dir = tmp_path / "checkpoint-1250"
    _make_adapter_dir(adapter_dir)
    manifest_path = tmp_path / "selected_adapter_manifest.json"

    manifest = qwen_qlora.freeze_selected_adapter(
        adapter_dir=adapter_dir,
        checkpoint_result=_checkpoint(1250, 0.4213),
        selection_file=tmp_path / "val_checkpoint_selection.json",
        manifest_path=manifest_path,
    )

    assert Path(manifest["adapter_model_safetensors"]["path"]) == (
        adapter_dir / "adapter_model.safetensors"
    )
    assert manifest["adapter_model_safetensors"]["sha256"] == sha256_file(
        adapter_dir / "adapter_model.safetensors"
    )
    assert manifest["selection_metric"] == "val_macro_f1"
    assert manifest["selected_global_step"] == 1250

    loaded = qwen_qlora.assert_adapter_frozen(manifest_path)
    assert loaded["selected_global_step"] == 1250


def test_assert_adapter_frozen_refuses_missing_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="selected_adapter_manifest"):
        qwen_qlora.assert_adapter_frozen(
            tmp_path / "selected_adapter_manifest.json"
        )


def test_assert_adapter_frozen_rejects_missing_adapter_files(tmp_path):
    adapter_dir = tmp_path / "checkpoint-1250"
    _make_adapter_dir(adapter_dir)
    manifest_path = tmp_path / "selected_adapter_manifest.json"
    qwen_qlora.freeze_selected_adapter(
        adapter_dir=adapter_dir,
        checkpoint_result=_checkpoint(1250, 0.4213),
        selection_file=tmp_path / "val_checkpoint_selection.json",
        manifest_path=manifest_path,
    )
    (adapter_dir / "adapter_model.safetensors").unlink()

    with pytest.raises(RuntimeError, match="adapter_model.safetensors"):
        qwen_qlora.assert_adapter_frozen(manifest_path)


def test_assert_adapter_frozen_rejects_hash_mismatch(tmp_path):
    adapter_dir = tmp_path / "checkpoint-1250"
    _make_adapter_dir(adapter_dir)
    manifest_path = tmp_path / "selected_adapter_manifest.json"
    qwen_qlora.freeze_selected_adapter(
        adapter_dir=adapter_dir,
        checkpoint_result=_checkpoint(1250, 0.4213),
        selection_file=tmp_path / "val_checkpoint_selection.json",
        manifest_path=manifest_path,
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        qwen_qlora.assert_adapter_frozen(manifest_path)


def test_formal_eval_is_gated_by_frozen_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="selected_adapter_manifest"):
        qwen_qlora.run_formal_eval(
            output_dir=tmp_path / "out",
            manifest_path=tmp_path / "selected_adapter_manifest.json",
        )


# ---------------------------------------------------------------------------
# TrainingArguments mapping from the frozen config
# ---------------------------------------------------------------------------


def test_training_arguments_apply_frozen_config():
    config = qwen_qlora.load_frozen_config()
    args = build_training_arguments(
        config, output_dir=Path("/tmp/formal-v1")
    )

    assert args.eval_strategy.value == "no"
    assert args.load_best_model_at_end is False
    assert args.metric_for_best_model is None
    assert args.greater_is_better is None
    assert args.save_strategy.value == "epoch"
    assert args.save_total_limit == 3
    assert args.num_train_epochs == 3
    assert args.learning_rate == 2e-5
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 16
    assert args.optim.value == "adamw_torch"
    assert args.lr_scheduler_type.value == "linear"
    assert args.warmup_ratio == 0.03
    assert args.bf16 is True
    assert args.gradient_checkpointing is True
    assert args.seed == 42
    assert args.data_seed == 42
