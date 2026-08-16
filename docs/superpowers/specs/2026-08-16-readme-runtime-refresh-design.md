# README 运行文档更新设计

## 目标

保持根目录 `README.md` 为唯一的中文项目入口，使新环境用户可按文档完成依赖安装、API 配置、MCP 测试、Agent 运行和可选的本地模型测试，不再依赖已过时的手工路径修改。

## 修订范围

- 仅修改根目录 `README.md`，不新增独立运行手册，不修改代码。
- 保留中文和现有简洁风格。
- 补全 `.venv` 的创建、激活、安装和验证命令。
- 明确命令的执行目录：依赖与测试在仓库根目录，Agent 从 `Financial-MCP-Agent/` 以 `python -m src.main` 启动。
- 说明 MCP 已通过 `sys.executable` 和仓库相对位置自动解析，无需手工编辑 `mcp_config.py`。
- 保持 `.env.example` 到 `.env` 的安全配置流程，不展示任何真实密钥。
- 明确 `USE_LOCAL_MODEL=local` 只替换 Summary Agent，其余四个 Agent 仍需 API；`FinR1/` 为可选本地模型路径。
- 将风险与情感训练描述更正为 4-bit QLoRA，并列出实际训练与测试脚本名。
- 增加现有 pytest、Baostock 和两个 adapter 测试命令，以及报告输出位置。
- 删除“必须手改 MCP 路径”、“当前是普通 LoRA”和“`requirements.txt` 尚未完整”等与当前仓库不符的断言。

## 文档结构

README 按以下顺序组织：

1. 项目概述与结构。
2. 环境准备与虚拟环境。
3. API 配置与本地模型边界。
4. MCP 配置原理和无需手工改路径的说明。
5. 分层测试命令。
6. 完整 Agent 启动和报告位置。
7. QLoRA 训练与 adapter 测试。
8. 常见问题和安全注意事项。

## 验证方式

- 检查 README 中的文件和目录均真实存在。
- 检查 README 不再包含已过时的 MCP、LoRA 和 requirements 描述。
- 在虚拟环境中执行 `python -m pip check`。
- 执行现有 pytest 套件，确认文档列出的主要测试入口有效。
- 不在文档修订中触发需要 API 费用或长时间 GPU 加载的全链路运行。

## 非目标

- 不下载 Fin-R1。
- 不重新训练 QLoRA adapter。
- 不修复中文新闻模型质量。
- 不更改 MCP、Agent 或测试代码。
