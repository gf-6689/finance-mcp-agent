# Finance MCP Agent

这是一个面向 A 股分析的多智能体项目。主程序使用 LangGraph 编排基本面、技术面、估值、新闻和汇总 Agent，并通过 stdio 连接本仓库中的 A 股 MCP 数据服务。

## 项目结构

```text
Finance/
├── Financial-MCP-Agent/          # 多智能体主程序和报告生成
├── a-share-mcp-is-just-i-need/   # A 股 MCP 数据服务
├── risk_nasdaq/                  # 风险分析训练数据
├── nasdaq_news_sentiment/        # 情感分析训练数据
├── train_qwen_risk.py            # 风险 LoRA 训练脚本
└── train_qwen_sentiment.py       # 情感 LoRA 训练脚本
```

## 推荐运行路线

第一次部署请使用 API 模式：

1. 创建 Python 3.12 环境并安装依赖。
2. 复制 `.env.example` 为 `Financial-MCP-Agent/.env`，填入真实的 OpenAI-compatible API 配置。
3. 将 `Financial-MCP-Agent/src/tools/mcp_config.py` 中的 MCP 启动命令改为服务器上的实际 Python 与 `mcp_server.py` 路径。
4. 先测试查询提取和 MCP 数据服务，再启动完整 Agent。

API 模式的核心环境变量：

```dotenv
OPENAI_COMPATIBLE_API_KEY=实际密钥
OPENAI_COMPATIBLE_BASE_URL=实际兼容地址
OPENAI_COMPATIBLE_MODEL=支持工具调用的模型名
USE_LOCAL_MODEL=api
```

不要将真实 `.env`、API Key、SSH 私钥、模型权重、日志或生成报告提交到 Git。

## 启动命令

完成服务器环境与 MCP 路径配置后，从主程序目录运行：

```bash
cd Financial-MCP-Agent
python -m src.main --command "帮我看看茅台(600519)这只股票值得投资吗"
```

生成的 Markdown 报告位于 `Financial-MCP-Agent/reports/`。

## 本地模型说明

- `USE_LOCAL_MODEL=local` 目前只影响 Summary Agent，其余 Agent 仍需要 API。
- Fin-R1、Qwen3-8B、LoRA adapter 和训练 checkpoint 不应提交到 GitHub。
- 当前风险与情感脚本是普通 LoRA，不是 4-bit QLoRA；24GB GPU 训练前需要进一步改造。
- 当前新闻分析会分别加载风险与情感模型。24GB GPU 不适合同时保留两个独立的 FP16 8B 基座。

## 当前已知部署注意事项

- MCP 子目录目前有 `uv.lock`，但没有 `pyproject.toml`；部署时应直接调用虚拟环境中的 Python 启动 `mcp_server.py`。
- 训练脚本中的模型路径和数据路径需要改成服务器实际路径。
- `requirements.txt` 尚未完整覆盖所有训练和 MCP 依赖，服务器部署时需要按部署指南补装。

