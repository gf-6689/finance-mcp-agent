import json
from pathlib import Path
import sys
from uuid import uuid4

import pytest


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from src.utils.execution_logger import ExecutionLogger
from src.utils.tool_logging_callback import ExecutionToolCallback, invoke_react_with_tool_logging


def read_tool_records(execution_logger, agent_name="news_agent"):
    path = execution_logger.execution_dir / "tools" / f"{agent_name}_tools.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_interleaved_tool_runs_are_logged_independently(tmp_path):
    execution_logger = ExecutionLogger(str(tmp_path))
    callback = ExecutionToolCallback("news_agent", execution_logger)
    first_run = uuid4()
    second_run = uuid4()

    callback.on_tool_start(
        {"name": "first_tool"}, "first input", run_id=first_run,
        inputs={"value": 1},
    )
    callback.on_tool_start(
        {"name": "second_tool"}, "second input", run_id=second_run,
        inputs={"value": 2},
    )
    callback.on_tool_end("second output", run_id=second_run)
    callback.on_tool_end("first output", run_id=first_run)

    records = read_tool_records(execution_logger)
    assert [record["tool_name"] for record in records] == [
        "second_tool", "first_tool",
    ]
    assert records[0]["input"] == {"value": 2}
    assert records[1]["input"] == {"value": 1}
    assert all(record["success"] is True for record in records)
    assert all(record["execution_time_seconds"] >= 0 for record in records)


def test_tool_error_is_logged_as_failure(tmp_path):
    execution_logger = ExecutionLogger(str(tmp_path))
    callback = ExecutionToolCallback("value_agent", execution_logger)
    run_id = uuid4()

    callback.on_tool_start({"name": "broken_tool"}, "{}", run_id=run_id)
    callback.on_tool_error(RuntimeError("boom"), run_id=run_id)

    records = read_tool_records(execution_logger, "value_agent")
    assert records[0]["success"] is False
    assert records[0]["error"] == "boom"


class FakeAsyncTool:
    name = "crawl_news"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, arguments):
        self.calls += 1
        return "scored news"


@pytest.mark.asyncio
async def test_explicit_async_invocation_is_logged_once(tmp_path):
    execution_logger = ExecutionLogger(str(tmp_path))
    callback = ExecutionToolCallback("news_agent", execution_logger)
    tool = FakeAsyncTool()

    output = await callback.ainvoke_tool(tool, {"query": "茅台", "top_k": 10})

    assert output == "scored news"
    assert tool.calls == 1
    records = read_tool_records(execution_logger)
    assert len(records) == 1
    assert records[0]["tool_name"] == "crawl_news"
    assert records[0]["input"] == {"query": "茅台", "top_k": 10}


class FakeReactAgent:
    def __init__(self):
        self.received = None

    async def ainvoke(self, input_data, config=None):
        self.received = (input_data, config)
        return {"messages": []}


@pytest.mark.asyncio
async def test_react_invocation_receives_logging_callback(tmp_path):
    execution_logger = ExecutionLogger(str(tmp_path))
    callback = ExecutionToolCallback("technical_agent", execution_logger)
    agent = FakeReactAgent()
    input_data = {"messages": ["question"]}

    result = await invoke_react_with_tool_logging(agent, input_data, callback)

    assert result == {"messages": []}
    assert agent.received == (input_data, {"callbacks": [callback]})
