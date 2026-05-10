# 更新日志

## [0.2.0] - 2026-05-10

### 新增
- LLM Agent 架构：接入大语言模型进行深度语义分析
- 多厂商支持：Claude / OpenAI / DeepSeek / 通义千问 / 智谱GLM / Kimi 等
- OpenAI 兼容模式：国产模型即插即用
- Token 概率检测：利用 LLM 的 logprobs 进行 AI 内容识别
- `.env` 环境变量配置系统
- `Makefile` 开发工具链
- 完整的开源社区文件（LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY）

### 架构
- 双模式引擎：LLM Agent 优先 + 统计方法降级
- `llm/` 包：抽象客户端层，统一多厂商 API
- `agents/` 包：LLM 驱动的语义 / 知识图谱 / AI检测 / 报告 Agent

## [0.1.0] - 2026-05-09

### 新增
- BERT 语义编码器（Sentence-BERT + FAISS）
- 知识图谱构建与匹配（NetworkX）
- AI 生成内容检测（统计特征 + 突发性分析）
- 学科配置系统（医学/计算机/人文/通用）
- 动态阈值调整模块
- 文本预处理流水线
- FastAPI REST API
- Redis 缓存层
- 多格式报告生成（HTML/Text/JSON）
- 147 个单元测试