from pathlib import Path
import sys
import time


AGENT_ROOT = Path(__file__).resolve().parents[1]
for module_name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
    sys.modules.pop(module_name)
sys.path.insert(0, str(AGENT_ROOT))

from src.main import complete_main_log


class FakeExecutionLogger:
    def __init__(self):
        self.calls = []

    def log_agent_complete(
        self, agent_name, output_data, execution_time, success=True, error=None
    ):
        self.calls.append(
            {
                "agent_name": agent_name,
                "output_data": output_data,
                "execution_time": execution_time,
                "success": success,
                "error": error,
            }
        )


def test_complete_main_log_records_success_and_report_path():
    execution_logger = FakeExecutionLogger()

    complete_main_log(
        execution_logger,
        started_at=time.monotonic() - 1,
        success=True,
        report_path="/tmp/report.md",
    )

    call = execution_logger.calls[0]
    assert call["agent_name"] == "main"
    assert call["success"] is True
    assert call["error"] is None
    assert call["execution_time"] > 0
    assert call["output_data"]["report_path"] == "/tmp/report.md"


def test_complete_main_log_records_failure_and_error():
    execution_logger = FakeExecutionLogger()

    complete_main_log(
        execution_logger,
        started_at=time.monotonic() - 1,
        success=False,
        error="workflow failed",
    )

    call = execution_logger.calls[0]
    assert call["agent_name"] == "main"
    assert call["success"] is False
    assert call["error"] == "workflow failed"
    assert call["execution_time"] > 0
