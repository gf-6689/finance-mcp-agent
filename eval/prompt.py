"""Frozen Risk classification prompt for the Qwen3-8B evaluations.

Protocol (version 1):

- task: predict the Risk label 1..5 from a news title and summary
- inputs: title and summary only (never Article, gold label, sentiment
  label or any other summary field)
- output: a single line FINAL_LABEL=X with X in {1, 2, 3, 4, 5}
- no JSON, no long explanations
- the chat template is always applied with enable_thinking=False

Both the Base and the future QLoRA evaluation must reuse this exact
module; the prompt may not be edited after the formal eval starts.
"""

RISK_PROMPT_VERSION = "1"

RISK_SYSTEM_MESSAGE = (
    "你是金融新闻风险分类器。给定一条金融新闻的标题和摘要，"
    "预测该新闻对相关股票的风险等级。\n"
    "风险等级定义（固定，共 5 级）：\n"
    "1 = 极低风险\n"
    "2 = 低风险\n"
    "3 = 中等风险\n"
    "4 = 高风险\n"
    "5 = 极高风险\n"
    "输出规则：只输出一行，格式必须严格为 FINAL_LABEL=X，"
    "其中 X 是 1 到 5 之间的整数。不要输出解释、推理过程或任何其他内容。"
)


def build_sample_text(title: str, summary: str) -> str:
    """Build the frozen title+summary feature string.

    Matches the TF-IDF + LR feature convention so every model sees
    exactly the same input content.
    """
    return title.strip() + "\n" + summary.strip()


def build_messages(
    title: str,
    summary: str,
) -> list[dict[str, str]]:
    """Build the frozen chat messages for one Risk sample."""
    return [
        {"role": "system", "content": RISK_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                f"新闻标题：{title}\n新闻摘要：{summary}"
            ),
        },
    ]


def apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    """Apply the Qwen3 chat template with thinking explicitly off.

    enable_thinking=False is part of the frozen protocol: this is a
    strict five-class offline evaluation that needs short parseable
    outputs, not reasoning traces. Base and QLoRA must both use this.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
