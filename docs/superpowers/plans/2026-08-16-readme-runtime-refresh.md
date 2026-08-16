# README Runtime Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 更新根目录中文 README，使新用户可使用虚拟环境、当前的动态 MCP 配置和正确脚本名完成安装、测试与运行。

**Architecture:** 仅修改 `README.md`，保留它作为唯一的中文项目入口。文档以“环境→配置→测试→运行→可选训练”为主线，并将 API 与本地模型的边界显式写入。

**Tech Stack:** Markdown、Python 3.12、venv/pip、pytest、LangGraph Agent、stdio MCP、QLoRA/PEFT

## Global Constraints

- 仅修改根目录 `README.md`，不新增独立运行手册，不修改代码。
- 保留中文和现有简洁风格。
- 不展示真实 API Key，不将当前机器状态描述为所有部署环境的保证。
- 不触发产生 API 费用或长时间 GPU 加载的全链路运行。

---

### Task 1: 校正并补全 README 运行说明

**Files:**
- Modify: `README.md`
- Reference: `Financial-MCP-Agent/.env.example`
- Reference: `Financial-MCP-Agent/src/tools/mcp_config.py`
- Reference: `train_qwen_risk.py`
- Reference: `train_qwen_sentiment.py`
- Reference: `test_risk_model.py`
- Reference: `test_qwen_sentiment.py`
- Reference: `a-share-mcp-is-just-i-need/test_baostock.py`

**Interfaces:**
- Consumes: 当前仓库目录、环境变量名称、脚本名和 MCP 启动方式。
- Produces: 一份可在新 Linux 环境中按顺序执行的中文 `README.md`。

- [ ] **Step 1: 记录修订前的静态检查结果**

Run:

```bash
rg -n "mcp_config.py.*改|\u666e\u901a LoRA|requirements.txt.*\u5c1a\u672a\u5b8c\u6574|source .venv/bin/activate|test_risk_model.py" README.md
```

Expected: 能匹配手工修改 MCP、普通 LoRA 和 requirements 不完整的旧描述；不匹配虚拟环境激活和正确风险测试脚本。

- [ ] **Step 2: 重写 README 的运行流程**

Write the following sections in `README.md`:

1. 项目结构，包含 Agent、MCP、数据、Qwen 基座和两个 adapter 目录。
2. 环境准备：从仓库根目录执行 `python3.12 -m venv .venv`、`source .venv/bin/activate`、`python -m pip install -U pip`、`python -m pip install -r requirements.txt`、`python -m pip check`。
3. API 配置：复制 `Financial-MCP-Agent/.env.example` 并设置四个环境变量，不写入真实值。
4. MCP 说明：`mcp_config.py` 使用当前解释器和仓库动态路径，无需手工修改。
5. 测试：在根目录执行 `python -m pytest tests Financial-MCP-Agent/tests -q`，可选执行 Baostock 和 adapter 测试脚本。
6. Agent 运行：进入 `Financial-MCP-Agent/` 后执行 `python -m src.main --command "帮我看看茅台(600519)这只股票值得投资吗"`，报告写入 `reports/`。
7. 本地模型：说明 `USE_LOCAL_MODEL=local` 仅影响 Summary Agent，`FinR1/` 不存在时继续使用 API 模式。
8. QLoRA：说明两个脚本使用 4-bit NF4 QLoRA，列出实际训练、输出和测试名称，提醒英文 Nasdaq 数据对中文新闻的局限。
9. 常见问题和安全说明：裸 Python 缺包、启动目录、Baostock 网络波动、敏感文件不提交。

- [ ] **Step 3: 验证 README 引用的本地路径**

Run:

```bash
for path in requirements.txt Financial-MCP-Agent/.env.example Financial-MCP-Agent/src/tools/mcp_config.py a-share-mcp-is-just-i-need/test_baostock.py train_qwen_risk.py train_qwen_sentiment.py test_risk_model.py test_qwen_sentiment.py; do test -e "$path" || exit 1; done
```

Expected: exit code 0.

- [ ] **Step 4: 确认过时描述已删除且必要命令已补全**

Run:

```bash
! rg -n "mcp_config.py.*改|\u666e\u901a LoRA|requirements.txt.*\u5c1a\u672a\u5b8c\u6574|test_qwen_risk.py" README.md
rg -n "source .venv/bin/activate|python -m src.main|test_risk_model.py|load_in_4bit|QLoRA|python -m pip check" README.md
```

Expected: 第一个搜索无匹配；第二个搜索的每个关键项都有匹配。

- [ ] **Step 5: 验证环境和现有测试**

Run:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest tests Financial-MCP-Agent/tests -q
git diff --check
```

Expected: pip reports `No broken requirements found.`; pytest reports 25 passed; `git diff --check` exits 0.

- [ ] **Step 6: 提交 README 修订**

```bash
git add README.md docs/superpowers/plans/2026-08-16-readme-runtime-refresh.md
git commit -m "docs: refresh setup and runtime guide"
```
