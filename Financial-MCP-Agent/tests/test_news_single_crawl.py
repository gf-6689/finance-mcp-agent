from pathlib import Path
import sys

import pytest


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from src.agents.news_agent import (
    build_news_agent_input,
    build_news_query,
    invoke_news_crawl_once,
    partition_news_tools,
)


class FakeTool:
    def __init__(self, name, result="unused"):
        self.name = name
        self.result = result
        self.calls = []

    async def ainvoke(self, arguments):
        self.calls.append(arguments)
        return self.result


class FakeToolCallback:
    def __init__(self):
        self.calls = 0

    async def ainvoke_tool(self, tool, arguments):
        self.calls += 1
        return await tool.ainvoke(arguments)


@pytest.mark.asyncio
async def test_news_crawl_is_invoked_once_and_removed_from_react_tools():
    crawl = FakeTool("crawl_news", result="已评分新闻")
    market = FakeTool("get_historical_k_data")
    callback = FakeToolCallback()

    selected, remaining = partition_news_tools([crawl, market])
    query = build_news_query("贵州茅台")
    result = await invoke_news_crawl_once(
        selected, query, top_k=10, tool_callback=callback
    )

    assert result == "已评分新闻"
    assert callback.calls == 1
    assert crawl.calls == [{"query": query, "top_k": 10}]
    assert [tool.name for tool in remaining] == ["get_historical_k_data"]


def test_news_prompt_embeds_crawled_result_without_requesting_another_crawl():
    prompt = build_news_agent_input(
        company_name="贵州茅台",
        stock_code="sh.600519",
        current_time_info="2026年08月16日 16:50",
        current_date="2026-08-16",
        crawled_news="新闻一：情感4，风险2",
    )

    assert "新闻一：情感4，风险2" in prompt
    assert "不要再次调用新闻爬取工具" in prompt


def test_missing_crawl_tool_fails_explicitly():
    with pytest.raises(RuntimeError, match="crawl_news"):
        partition_news_tools([FakeTool("get_stock_basic_info")])


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["", "爬取新闻时出错: timeout"])
async def test_empty_or_error_crawl_result_fails_without_retry(result):
    crawl = FakeTool("crawl_news", result=result)

    with pytest.raises(RuntimeError, match="crawl_news"):
        await invoke_news_crawl_once(crawl, "贵州茅台", top_k=10)

    assert len(crawl.calls) == 1
