# 稳定性回归修复实施计划

> **供智能体执行者使用：** 必须按任务逐项执行；建议使用 `superpowers:subagent-driven-development`，或使用 `superpowers:executing-plans`。所有步骤使用复选框跟踪。

**目标：** 修复月线字段错误、新闻模型跨进程重复加载、工具计数为零和 Main 状态误报，并以一次无真实错误的完整回归证明修复。

**架构：** 在数据源边界按频率验证 K 线字段；由新闻 Agent 在 ReAct 前直接且仅调用一次 `crawl_news`；使用 LangChain 回调把实际工具调用写入现有 JSONL；由 Main 在最终汇总前显式结束自己的执行记录。所有行为保持现有公开 MCP 接口兼容。

**技术栈：** Python 3.12、pytest、LangChain/LangGraph、MCP、Baostock、Transformers、PEFT、bitsandbytes。

## 全局约束

- 修改前恢复点必须保持为远端分支 `backup/pre-stability-fixes-20260816`，提交 `e8b7e94e0469ac997ccbc4185e52570d37713249`。
- 开发分支为 `fix/stability-regression`，在隔离 worktree 中执行。
- 生产代码修改必须先有能够正确失败的测试。
- 不提交模型、数据集、日志、报告、checkpoint 或 `.env`。
- 不连续重试 Baostock；完整回归只运行一次。
- 只有全部验收条件通过后才推送修复分支并更新 `main`。

---

### 任务一：建立隔离工作区并确认基线

**文件：**
- 检查：`.gitignore`
- 检查：现有测试文件

**接口：**
- 输入：当前分支 `fix/stability-regression`
- 输出：隔离 worktree 与可重复的基线测试结果

- [ ] **步骤 1：检测当前 Git 隔离状态**

运行：

```bash
GIT_DIR=$(cd "$(git rev-parse --git-dir)" && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
BRANCH=$(git branch --show-current)
printf 'GIT_DIR=%s\nGIT_COMMON=%s\nBRANCH=%s\n' "$GIT_DIR" "$GIT_COMMON" "$BRANCH"
```

- [ ] **步骤 2：确认 `.worktrees` 被忽略并创建隔离分支**

```bash
git check-ignore -q .worktrees
git switch main
git worktree add .worktrees/stability-regression -b fix/stability-regression-impl fix/stability-regression
```

如果 `.worktrees` 尚未忽略，先把 `/.worktrees/` 加入 `.gitignore`，单独提交后再创建。

- [ ] **步骤 3：运行基线检查**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m compileall -q Financial-MCP-Agent/src a-share-mcp-is-just-i-need/src
/root/autodl-tmp/Finance/.venv/bin/python -m pytest -q
```

记录现有测试失败；若失败与本次范围相关，先停止实施并定位。

---

### 任务二：按频率解析和验证 K 线字段

**文件：**
- 修改：`a-share-mcp-is-just-i-need/src/baostock_data_source.py`
- 新建：`tests/test_kline_fields.py`

**接口：**
- 生成：`resolve_k_fields(frequency: str, fields: Optional[List[str]]) -> str`
- 调用方：`BaostockDataSource.get_historical_k_data`

- [ ] **步骤 1：编写失败测试**

测试必须断言：

```python
def test_monthly_defaults_exclude_daily_only_fields():
    resolved = resolve_k_fields("m", None).split(",")
    assert "close" in resolved
    assert "preclose" not in resolved
    assert "peTTM" not in resolved

def test_invalid_explicit_monthly_fields_fail_locally():
    with pytest.raises(ValueError, match="peTTM"):
        resolve_k_fields("m", ["date", "close", "peTTM"])

def test_daily_defaults_keep_valuation_fields():
    resolved = resolve_k_fields("d", None).split(",")
    assert "preclose" in resolved
    assert "peTTM" in resolved
```

- [ ] **步骤 2：验证测试因缺少解析函数而失败**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m pytest tests/test_kline_fields.py -q
```

预期：失败原因是 `resolve_k_fields` 尚不存在，不是导入路径错误。

- [ ] **步骤 3：实现最小字段解析逻辑**

在数据源模块增加：

```python
AGGREGATE_K_FIELDS = [
    "date", "code", "open", "high", "low", "close",
    "volume", "amount", "adjustflag",
]

def resolve_k_fields(frequency: str, fields: Optional[List[str]]) -> str:
    allowed = AGGREGATE_K_FIELDS if frequency in {"w", "m"} else DEFAULT_K_FIELDS
    selected = fields or allowed
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(
            f"Unsupported fields for frequency '{frequency}': {', '.join(invalid)}"
        )
    return format_fields(selected, allowed)
```

并把 `get_historical_k_data` 中 `_format_fields(fields, DEFAULT_K_FIELDS)` 改为 `resolve_k_fields(frequency, fields)`。

- [ ] **步骤 4：运行测试并确认通过**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m pytest tests/test_kline_fields.py -q
```

- [ ] **步骤 5：提交任务一成果**

```bash
git add tests/test_kline_fields.py a-share-mcp-is-just-i-need/src/baostock_data_source.py
git commit -m "fix: validate K-line fields by frequency"
```

---

### 任务三：新闻 Agent 单次抓取

**文件：**
- 修改：`Financial-MCP-Agent/src/agents/news_agent.py`
- 新建：`Financial-MCP-Agent/tests/test_news_single_crawl.py`

**接口：**
- 生成：`partition_news_tools(tools) -> tuple[Any, list[Any]]`
- 生成：`build_news_query(company_name: str) -> str`
- 生成：`invoke_news_crawl_once(tool, query: str, top_k: int, execution_logger, agent_name: str) -> str`

- [ ] **步骤 1：编写失败测试**

使用记录调用次数的异步假工具，验证：

```python
@pytest.mark.asyncio
async def test_news_crawl_is_invoked_once_and_removed_from_react_tools():
    crawl = FakeTool("crawl_news", result="已评分新闻")
    market = FakeTool("get_historical_k_data")
    selected, remaining = partition_news_tools([crawl, market])
    result = await invoke_news_crawl_once(
        selected, "贵州茅台 最新新闻 股价 业绩 白酒行业动态", 10,
        execution_logger, "news_agent",
    )
    assert result == "已评分新闻"
    assert crawl.calls == 1
    assert [tool.name for tool in remaining] == ["get_historical_k_data"]
```

另写测试验证缺失工具、空结果和以“爬取新闻时出错”开头的结果会明确失败。

- [ ] **步骤 2：运行并观察正确失败**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m pytest Financial-MCP-Agent/tests/test_news_single_crawl.py -q
```

- [ ] **步骤 3：实现数据获取与综合分析分离**

在 `news_agent` 中：

```python
crawl_tool, react_tools = partition_news_tools(mcp_tools)
news_query = build_news_query(company_name)
crawled_news = await invoke_news_crawl_once(
    crawl_tool, news_query, 10, execution_logger, agent_name
)
agent = create_react_agent(llm, react_tools)
```

将 `crawled_news` 原文嵌入 `agent_input`，明确要求只基于该新闻数据评分，不再调用新闻爬取工具。

- [ ] **步骤 4：运行测试并确认通过**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m pytest Financial-MCP-Agent/tests/test_news_single_crawl.py -q
```

- [ ] **步骤 5：提交单次抓取修复**

```bash
git add Financial-MCP-Agent/src/agents/news_agent.py Financial-MCP-Agent/tests/test_news_single_crawl.py
git commit -m "fix: crawl news once per analysis"
```

---

### 任务四：记录真实工具调用

**文件：**
- 新建：`Financial-MCP-Agent/src/utils/tool_logging_callback.py`
- 新建：`Financial-MCP-Agent/tests/test_tool_logging_callback.py`
- 修改：`Financial-MCP-Agent/src/agents/fundamental_agent.py`
- 修改：`Financial-MCP-Agent/src/agents/technical_agent.py`
- 修改：`Financial-MCP-Agent/src/agents/value_agent.py`
- 修改：`Financial-MCP-Agent/src/agents/news_agent.py`

**接口：**
- 生成：`ExecutionToolCallback(BaseCallbackHandler)`
- 构造：`ExecutionToolCallback(agent_name: str, execution_logger: ExecutionLogger)`

- [ ] **步骤 1：编写并发回调失败测试**

测试两个不同 `run_id` 交错开始和结束，断言产生两条工具 JSONL 记录、名称和耗时互不覆盖；再测试 `on_tool_error` 写入 `success=False`。

- [ ] **步骤 2：运行并观察回调类不存在的失败**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m pytest Financial-MCP-Agent/tests/test_tool_logging_callback.py -q
```

- [ ] **步骤 3：实现最小回调处理器**

核心状态：

```python
self._starts: dict[UUID, tuple[float, str, dict]] = {}
```

`on_tool_start` 从 `serialized["name"]` 读取工具名，并按 `run_id` 保存开始数据；结束或错误时 `pop(run_id)`，调用现有 `execution_logger.log_tool_usage(...)`。日志写入异常只记 warning。

- [ ] **步骤 4：把回调接入四个 ReAct Agent**

每个 Agent 调用：

```python
tool_callback = ExecutionToolCallback(agent_name, execution_logger)
response = await agent.ainvoke(
    input_data,
    config={"callbacks": [tool_callback]},
)
```

新闻的直接抓取使用同一回调的显式辅助方法记录，避免重复记账。

- [ ] **步骤 5：运行回调与新闻测试**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m pytest \
  Financial-MCP-Agent/tests/test_tool_logging_callback.py \
  Financial-MCP-Agent/tests/test_news_single_crawl.py -q
```

- [ ] **步骤 6：提交工具日志修复**

```bash
git add Financial-MCP-Agent/src/utils/tool_logging_callback.py \
  Financial-MCP-Agent/src/agents/fundamental_agent.py \
  Financial-MCP-Agent/src/agents/technical_agent.py \
  Financial-MCP-Agent/src/agents/value_agent.py \
  Financial-MCP-Agent/src/agents/news_agent.py \
  Financial-MCP-Agent/tests/test_tool_logging_callback.py
git commit -m "fix: record MCP tool usage"
```

---

### 任务五：正确结束 Main 执行记录

**文件：**
- 修改：`Financial-MCP-Agent/src/main.py`
- 新建：`Financial-MCP-Agent/tests/test_main_execution_logging.py`

**接口：**
- 生成：`complete_main_log(execution_logger, started_at: float, success: bool, report_path: Optional[str] = None, error: Optional[str] = None) -> None`

- [ ] **步骤 1：编写成功和失败路径测试**

成功断言 `main_execution.json` 包含 `success: true`、非零耗时和报告路径；失败断言包含 `success: false` 与错误文本。

- [ ] **步骤 2：运行并观察辅助函数不存在的失败**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m pytest Financial-MCP-Agent/tests/test_main_execution_logging.py -q
```

- [ ] **步骤 3：实现并接入 Main 生命周期**

在解析完用户输入后保存 `main_started_at = time.time()`。成功路径在 `finalize_execution_logger` 前完成 Main；异常路径也先完成 Main，再保存 `execution_dir`，最后 finalize，避免 finalize 后重新创建新的全局日志实例。

- [ ] **步骤 4：运行 Main 日志与摘要测试**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m pytest Financial-MCP-Agent/tests/test_main_execution_logging.py -q
```

- [ ] **步骤 5：提交 Main 日志修复**

```bash
git add Financial-MCP-Agent/src/main.py Financial-MCP-Agent/tests/test_main_execution_logging.py
git commit -m "fix: finalize main execution logging"
```

---

### 任务六：针对性验证和最小在线检查

**文件：**
- 验证：全部修改文件

- [ ] **步骤 1：编译修改模块**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m compileall -q Financial-MCP-Agent/src a-share-mcp-is-just-i-need/src tests Financial-MCP-Agent/tests
```

- [ ] **步骤 2：运行全部新增测试**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m pytest \
  tests/test_kline_fields.py \
  Financial-MCP-Agent/tests/test_news_single_crawl.py \
  Financial-MCP-Agent/tests/test_tool_logging_callback.py \
  Financial-MCP-Agent/tests/test_main_execution_logging.py -q
```

- [ ] **步骤 3：执行一次 Baostock 最小日线/月线验证**

同一 Python 进程只登录一次，分别请求日线默认字段和月线聚合字段，要求两者 `error_code == "0"` 且不出现 `10004012`，然后正常登出。

- [ ] **步骤 4：检查差异和仓库安全**

```bash
git diff --check
git status --short
git diff --stat fix/stability-regression...HEAD
```

---

### 任务七：完整回归、最终提交与推送

**文件：**
- 生成但不提交：`Financial-MCP-Agent/logs/`
- 生成但不提交：`Financial-MCP-Agent/reports/`

- [ ] **步骤 1：确认没有残留进程和 GPU 占用**

```bash
pgrep -af 'src.main|mcp_server.py|test_qwen|train_qwen' || true
nvidia-smi
```

- [ ] **步骤 2：只运行一次完整回归**

```bash
cd Financial-MCP-Agent
set -o pipefail
RUN_LOG="/tmp/finance_stability_regression_$(date +%Y%m%d_%H%M%S).log"
/root/autodl-tmp/Finance/.venv/bin/python -u -m src.main \
  --command "帮我看看茅台(600519)这只股票值得投资吗" \
  2>&1 | tee "$RUN_LOG"
test "${PIPESTATUS[0]}" -eq 0
```

- [ ] **步骤 3：核验完整回归证据**

必须验证：报告大于 10 KB；整体、Main 和五个业务 Agent 全部成功；工具 JSONL 条数等于 `tools_used_count`；`crawl_news` 只有一次；风险和情感模型各成功加载一次；不存在 `10004012`、模型加载错误、真实 `ERROR` 或 traceback。

- [ ] **步骤 4：运行最终测试和 Git 检查**

```bash
/root/autodl-tmp/Finance/.venv/bin/python -m pytest tests Financial-MCP-Agent/tests -q
git diff --check
git status --short --branch
```

- [ ] **步骤 5：确认最终验证没有产生文件修改**

```bash
git status --short
```

若有源代码或测试文件变化，停止最终流程，返回对应任务重新执行“失败测试 → 最小实现 → 通过测试 → 独立提交”，不得在最终步骤临时混入未验证修改。

- [ ] **步骤 6：推送修复分支**

```bash
git push -u origin fix/stability-regression-impl
```

- [ ] **步骤 7：在确认全部证据后更新 Main**

只有用户确认后，才把已验证提交快进合并或拣选到 `main` 并推送。任何失败均保留 `main` 和远端恢复分支不变。
