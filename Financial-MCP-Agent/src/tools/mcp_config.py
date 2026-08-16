"""A股 MCP 服务配置。"""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MCP_SERVER_PATH = REPOSITORY_ROOT / "a-share-mcp-is-just-i-need" / "mcp_server.py"

SERVER_CONFIGS = {
    "a_share_mcp_v2": {
        "command": sys.executable,
        "args": [
            "-u",
            str(MCP_SERVER_PATH),
        ],
        "transport": "stdio",
    }
}