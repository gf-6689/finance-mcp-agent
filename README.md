# Finance MCP Agent

这是一个面向 A 股分析的多智能体项目。主程序使用 LangGraph 编排基本面、技术面、估值、新闻和汇总 Agent，并通过 stdio 连接本仓库中的 A 股 MCP 数据服务。

## 项目结构

```text
Finance/
├── Financial-MCP-Agent/          # 多智能体主程序和报告生成
├── a-share-mcp-is-just-i-need/   # A 股 MCP 数据服务
├── risk_nasdaq/                  # 风险分析训练数据
├── nasdaq_news_sentiment/        # 情感分析训练数据
├── Qwen/                         # Qwen3-8B 基座模型（本地可选）
├── qwen_risk_model/              # 风险 QLoRA adapter
├── qwen_sentiment_model/         # 情感 QLoRA adapter
├── train_qwen_risk.py            # 风险 QLoRA 训练脚本
└── train_qwen_sentiment.py       # 情感 QLoRA 训练脚本
```

Qwen、adapter、checkpoint 和训练数据体积较大，可按是否需要本地新闻分析选择性准备。API 模式不需要 Fin-R1。

## 1. 准备环境

以下命令默认在仓库根目录 `Finance/` 执行。建议使用 Python 3.12 和独立虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

每次新开终端后，都要先在仓库根目录激活虚拟环境：

```bash
source .venv/bin/activate
```

不要直接使用未安装项目依赖的系统 Python 或 Conda 基础环境，否则可能出现 `ModuleNotFoundError`。

## 2. 配置 API

首次部署建议先使用 API 模式跑通全链路：

```bash
cp Financial-MCP-Agent/.env.example Financial-MCP-Agent/.env
```

然后编辑 `Financial-MCP-Agent/.env`：

```dotenv
OPENAI_COMPATIBLE_API_KEY=实际密钥
OPENAI_COMPATIBLE_BASE_URL=实际 OpenAI-compatible API 地址
OPENAI_COMPATIBLE_MODEL=支持工具调用的模型名
USE_LOCAL_MODEL=api
```

API 模式下五个 Agent 都使用上述接口。项目可配置 DeepSeek 或其他兼容服务，但所选模型需要支持项目使用的工具调用方式。

## 3. MCP 配置

无需手工修改 `Financial-MCP-Agent/src/tools/mcp_config.py`。当前配置会：

- 使用 `sys.executable` 启动 MCP，复用当前已激活的 `.venv` Python；
- 根据配置文件位置动态推导仓库根目录；
- 直接启动 `a-share-mcp-is-just-i-need/mcp_server.py`。

MCP 子目录只有 `uv.lock` 而没有 `pyproject.toml`，因此不要使用旧版文档中的 `uv run --directory ...` 配置。

## 4. 运行测试

### 仓库回归测试

在仓库根目录执行：

```bash
python -m pytest tests Financial-MCP-Agent/tests -q
```

这些测试覆盖 K 线字段、财报期选择、MCP 路径、新闻抓取去重、工具日志和主程序执行日志。

### Baostock/MCP 数据测试

```bash
python a-share-mcp-is-just-i-need/test_baostock.py
```

该测试需要联网访问 Baostock，可能因限流、黑名单或网络波动失败。网络类失败不代表本地依赖或 MCP 路径一定存在问题。

### QLoRA adapter 测试

已准备 Qwen 基座模型和两个 adapter 时，可执行：

```bash
python test_risk_model.py
python test_qwen_sentiment.py
```

这两个脚本会加载大模型并占用 GPU 显存，不属于快速 pytest 回归测试。

## 5. 启动完整 Agent

保持 `.venv` 处于激活状态，然后从主程序目录以模块方式启动：

```bash
cd Financial-MCP-Agent
python -m src.main --command "帮我看看茅台(600519)这只股票值得投资吗"
```

内部导入使用 `from src...`，因此建议保持上述启动目录和 `python -m src.main` 形式，无需额外设置 `PYTHONPATH`。

生成的 Markdown 报告位于：

```text
Financial-MCP-Agent/reports/
```

完整运行会调用外部 API 和联网数据源，需要考虑 API 费用、账户额度和数据源可用性。

## 6. 本地 Fin-R1 汇总模型（可选）

如果已将 Fin-R1 放在仓库根目录的 `FinR1/` 中，可将 `.env` 改为：

```dotenv
USE_LOCAL_MODEL=local
```

该选项只会让 Summary Agent 使用本地 Fin-R1。基本面、技术面、估值和新闻四个 Agent 仍使用 OpenAI-compatible API，因此仍须保留有效的 API 配置。

没有下载 Fin-R1 时，请保持 `USE_LOCAL_MODEL=api`。

## 7. QLoRA 训练说明（可选）

风险与情感训练脚本当前使用 4-bit QLoRA，量化配置为 `load_in_4bit=True`、NF4、bfloat16 计算和 double quant。

| 用途 | 训练脚本 | 输出目录 | 测试脚本 |
|---|---|---|---|
| 风险分析 | `train_qwen_risk.py` | `qwen_risk_model/` | `test_risk_model.py` |
| 情感分析 | `train_qwen_sentiment.py` | `qwen_sentiment_model/` | `test_qwen_sentiment.py` |

两个训练脚本默认从 `/root/autodl-tmp/Finance/Qwen` 加载基座模型，数据分别来自 `risk_nasdaq/` 和 `nasdaq_news_sentiment/`。如果仓库部署在其他位置，训练前需要更改训练脚本中的本地模型路径。

训练数据主要是英文 Nasdaq 新闻。本地模型分析中文新闻时可能输出不稳定或触发“无法分析”兜底，现场演示前建议使用中文样本单独验证。

### 新闻标签定义

以下标签语义适用于数据清洗、训练、验证、测试、模型 Prompt、人工抽查和最终评测，不得在不同阶段修改。

#### 风险标签 `risk_deepseek`

风险等级表示新闻对指定股票投资风险的影响程度：

| 标签 | 含义 |
|---:|---|
| 1 | 极低风险（very low risk） |
| 2 | 低风险（low risk） |
| 3 | 中等风险（moderate risk）；新闻没有明确风险信号时默认使用该等级 |
| 4 | 高风险（high risk） |
| 5 | 极高风险（very high risk） |

#### 情感标签 `sentiment_deepseek`

情感等级表示新闻对指定股票的情感倾向：

| 标签 | 含义 |
|---:|---|
| 1 | 负面（negative） |
| 2 | 轻微负面（somewhat negative） |
| 3 | 中性（neutral） |
| 4 | 轻微正面（somewhat positive） |
| 5 | 正面（positive） |

上述数据来自纳斯达克英文新闻；这些标签不代表 A 股新闻标签，也不构成投资建议。

## 安全与部署注意事项

- 不要将真实 `.env`、API Key、SSH 私钥、日志、生成报告、数据集或模型权重提交到 Git。
- 复制仓库后，应在新虚拟环境中安装依赖并执行 `python -m pip check`。当前机器通过检查不代表任意平台与 Python 版本都一定兼容。
- RTX 50 系列或其他新 GPU 需要使用与 CUDA 驱动匹配的 PyTorch、bitsandbytes 和 Transformers 版本。
- Baostock 和新闻网页是外部数据源，结果会受交易日、公告披露、限流和页面变更影响。
