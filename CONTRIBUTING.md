# 贡献指南

感谢你对语义级论文查重系统的关注！以下是参与贡献的指引。

## 开发环境设置

```bash
git clone <your-fork-url>
cd thesis-plagiarism-check
pip install -e ".[dev]"
```

## 代码规范

- 使用 `black` 格式化（line-length=100）
- 使用 `ruff` 做 linter
- 类型注解完整，通过 `mypy` 检查
- 所有函数/类需要中文或英文 docstring

## 提交 PR 前检查

```bash
make format     # 格式化代码
make lint       # 代码检查
make test       # 跑全部测试
```

## 新增 LLM 厂商

1. 在 `src/thesischeck/llm/openai_style.py` 的 `PROVIDER_CONFIGS` 中添加厂商配置
2. 在 `llm/factory.py` 的 `known_providers` 中添加厂商名
3. 更新 `.env.example` 中的环境变量

## 测试要求

- 统计路径的测试不需要 API key
- LLM 相关的测试标记 `@pytest.mark.llm`
- 新功能需要对应的单元测试