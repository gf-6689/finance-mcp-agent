"""把 LangChain 工具事件写入项目现有执行日志。"""

import logging
import threading
import time
from typing import Any
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler


logger = logging.getLogger(__name__)


class ExecutionToolCallback(BaseCallbackHandler):
    """按 run_id 跟踪并发工具调用，并持久化真实调用记录。"""

    def __init__(self, agent_name: str, execution_logger):
        self.agent_name = agent_name
        self.execution_logger = execution_logger
        self._starts: dict[UUID, tuple[float, str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name") or "unknown_tool"
        tool_input = inputs if isinstance(inputs, dict) else {"input": input_str}
        with self._lock:
            self._starts[run_id] = (time.monotonic(), tool_name, tool_input)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, output=output, success=True, error=None)

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._finish(run_id, output="", success=False, error=str(error))

    def _finish(
        self,
        run_id: UUID,
        *,
        output: Any,
        success: bool,
        error: str | None,
    ) -> None:
        with self._lock:
            start_data = self._starts.pop(run_id, None)
        if start_data is None:
            logger.warning("Missing tool start event for run_id=%s", run_id)
            return

        started_at, tool_name, tool_input = start_data
        try:
            self.execution_logger.log_tool_usage(
                agent_name=self.agent_name,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=output,
                execution_time=max(0.0, time.monotonic() - started_at),
                success=success,
                error=error,
            )
        except Exception as log_error:
            logger.warning("Failed to persist tool usage: %s", log_error)

    async def ainvoke_tool(self, tool, arguments: dict[str, Any]) -> Any:
        """执行一个显式异步工具调用，并复用相同日志事件语义。"""
        run_id = uuid4()
        self.on_tool_start(
            {"name": getattr(tool, "name", "unknown_tool")},
            str(arguments),
            run_id=run_id,
            inputs=arguments,
        )
        try:
            output = await tool.ainvoke(arguments)
        except Exception as error:
            self.on_tool_error(error, run_id=run_id)
            raise
        self.on_tool_end(output, run_id=run_id)
        return output


async def invoke_react_with_tool_logging(agent, input_data, callback):
    """以统一配置调用 ReAct Agent，确保其工具事件进入执行日志。"""
    return await agent.ainvoke(input_data, config={"callbacks": [callback]})
