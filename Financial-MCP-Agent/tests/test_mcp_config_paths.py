import sys
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent
sys.path.insert(0, str(AGENT_ROOT))


def test_mcp_config_uses_current_python_and_repository_server():
    from src.tools.mcp_config import SERVER_CONFIGS

    config = SERVER_CONFIGS["a_share_mcp_v2"]

    assert config["command"] == sys.executable
    assert Path(config["args"][-1]).resolve() == (
        REPO_ROOT / "a-share-mcp-is-just-i-need" / "mcp_server.py"
    ).resolve()
    assert Path(config["args"][-1]).is_file()
