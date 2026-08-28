from eval.prompt import (
    RISK_PROMPT_VERSION,
    RISK_SYSTEM_MESSAGE,
    apply_chat_template,
    build_messages,
    build_sample_text,
)


class _RecordingTokenizer:
    """Fake tokenizer that records the chat template call arguments."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "<|template|>"


def test_prompt_version_is_frozen():
    assert RISK_PROMPT_VERSION == "1"


def test_system_message_defines_all_five_labels():
    for label in range(1, 6):
        assert f"{label} =" in RISK_SYSTEM_MESSAGE


def test_system_message_requires_final_label_format():
    assert "FINAL_LABEL=X" in RISK_SYSTEM_MESSAGE


def test_build_messages_contains_only_title_and_summary():
    messages = build_messages("苹果发布新机", "销量增长 20%")

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == RISK_SYSTEM_MESSAGE
    assert messages[1]["role"] == "user"
    assert "苹果发布新机" in messages[1]["content"]
    assert "销量增长 20%" in messages[1]["content"]


def test_build_sample_text_matches_lr_feature_convention():
    assert build_sample_text(" 标题 ", " 摘要 ") == "标题\n摘要"


def test_chat_template_forces_enable_thinking_false():
    tokenizer = _RecordingTokenizer()
    messages = build_messages("标题", "摘要")

    apply_chat_template(tokenizer, messages)

    (_recorded_messages, kwargs) = tokenizer.calls[0]
    assert kwargs["enable_thinking"] is False
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is False
