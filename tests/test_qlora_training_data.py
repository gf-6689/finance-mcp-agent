import torch
import pytest

from eval import qlora_smoke


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "PROMPT:" + messages[-1]["content"] + "\nASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text.endswith(self.eos_token):
            content = text[: -len(self.eos_token)]
            return [ord(char) + 10 for char in content] + [self.eos_token_id]
        return [ord(char) + 10 for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        pieces = []
        for token_id in token_ids:
            if token_id == self.eos_token_id:
                if not skip_special_tokens:
                    pieces.append(self.eos_token)
            elif token_id != self.pad_token_id:
                pieces.append(chr(token_id - 10))
        return "".join(pieces)


def _row(label=3, title="Title", summary="Summary"):
    return {
        "sample_id": "sample-1",
        "title": title,
        "summary": summary,
        "label": str(label),
    }


def test_target_only_collator_preserves_mask_and_masks_padding():
    tokenizer = TinyTokenizer()
    collator = qlora_smoke.TargetOnlyDataCollator(tokenizer)
    examples = [
        {
            "input_ids": [11, 12, 13, 14],
            "labels": [-100, -100, 13, 14],
        },
        {
            "input_ids": [21, 22],
            "labels": [-100, 22],
        },
    ]

    batch = collator(examples)

    assert batch["input_ids"].tolist() == [
        [11, 12, 13, 14],
        [0, 0, 21, 22],
    ]
    assert batch["labels"].tolist() == [
        [-100, -100, 13, 14],
        [-100, -100, -100, 22],
    ]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 1],
        [0, 0, 1, 1],
    ]
    for input_ids, labels, attention_mask in zip(
        batch["input_ids"], batch["labels"], batch["attention_mask"]
    ):
        assert len(input_ids) == len(labels) == len(attention_mask)


def test_training_sequence_masks_prompt_and_keeps_complete_target():
    tokenizer = TinyTokenizer()

    sequence = qlora_smoke.build_training_sequences(
        [_row(label=4)], tokenizer, max_seq_length=256
    )[0]

    supervised_ids = [
        token_id
        for token_id, label in zip(sequence["input_ids"], sequence["labels"])
        if label != -100
    ]
    supervised_labels = [
        label for label in sequence["labels"] if label != -100
    ]
    assert supervised_ids == supervised_labels
    assert tokenizer.decode(supervised_labels) == "FINAL_LABEL=4<eos>"
    assert all(
        label == -100
        for label in sequence["labels"][: -len(supervised_labels)]
    )


def test_long_prompt_is_truncated_without_truncating_target():
    tokenizer = TinyTokenizer()
    target_ids = tokenizer.encode(
        "FINAL_LABEL=5" + tokenizer.eos_token,
        add_special_tokens=False,
    )
    max_seq_length = len(target_ids) + 7

    sequence = qlora_smoke.build_training_sequences(
        [_row(label=5, title="T" * 500, summary="S" * 500)],
        tokenizer,
        max_seq_length=max_seq_length,
    )[0]

    assert len(sequence["input_ids"]) == max_seq_length
    assert len(sequence["labels"]) == max_seq_length
    assert sequence["input_ids"][-len(target_ids):] == target_ids
    assert sequence["labels"][-len(target_ids):] == target_ids
    assert tokenizer.decode(sequence["labels"][-len(target_ids):]) == (
        "FINAL_LABEL=5<eos>"
    )
    assert any(label != -100 for label in sequence["labels"])


def test_training_sequence_rejects_length_that_cannot_fit_target():
    tokenizer = TinyTokenizer()
    target_ids = tokenizer.encode(
        "FINAL_LABEL=3" + tokenizer.eos_token,
        add_special_tokens=False,
    )

    with pytest.raises(ValueError, match="cannot fit the complete target"):
        qlora_smoke.build_training_sequences(
            [_row(label=3)],
            tokenizer,
            max_seq_length=len(target_ids) - 1,
        )
