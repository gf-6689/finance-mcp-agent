"""A股 MCP 服务配置。"""

SERVER_CONFIGS = {
    "a_share_mcp_v2": {
        "command": "/root/autodl-tmp/Finance/.venv/bin/python",
        "args": [
            "-u",
            "/root/autodl-tmp/Finance/a-share-mcp-is-just-i-need/mcp_server.py",
        ],
        "transport": "stdio",
    }
}