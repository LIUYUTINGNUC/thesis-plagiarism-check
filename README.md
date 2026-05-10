<div align="center">

# 语义级论文查重系统

**Semantic Thesis Plagiarism Detection System**

[![CI](https://github.com/LIUYUTINGNUC/thesis-plagiarism-check/actions/workflows/ci.yml/badge.svg)](https://github.com/LIUYUTINGNUC/thesis-plagiarism-check/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**LLM Agent + BERT 语义编码 + 知识图谱 + AI 内容检测** — 智能论文查重引擎，**自由对接任意 LLM API**

</div>

---

## 📋 目录 | Table of Contents

- [项目简介](#-项目简介--introduction)
- [核心特性](#-核心特性--features)
- [支持的 LLM 厂商](#-支持的-llm-厂商--supported-llm-providers)
- [架构总览](#-架构总览--architecture)
- [快速开始](#-快速开始--quick-start)
- [API 文档](#-api-文档--api-documentation)
- [配置学科](#-配置学科--discipline-configuration)
- [技术栈](#-技术栈--tech-stack)
- [贡献指南](#-贡献指南--contributing)
- [许可证](#-许可证--license)

---

## 📖 项目简介 | Introduction

### 中文

**ThesisCheck** 是一个面向学术场景的语义级论文查重系统，不同于传统基于字符串匹配的查重工具，它从**语义理解**层面检测学术不端行为。

系统采用 **LLM Agent 双模式架构**：
- **主模式：LLM Agent 模式** — 接入大语言模型进行深度语义分析、实体关系提取和 AI 内容检测
- **降级模式：统计模式** — 使用 Sentence-BERT + FAISS + 知识图谱的纯统计方法，无需任何 API Key

系统不绑定任何特定厂商，提供两种接口类型：
- **Claude 原生接口** — Anthropic Claude 专用（原生 SDK，支持 token 概率检测）
- **OpenAI 兼容接口** — 可对接任意兼容 OpenAI Chat Completions 格式的 API（DeepSeek、通义千问、GLM 等），用户自由配置 base_url、model 和 api_key

配置任意 API 即可启用 LLM Agent 模式。

### English

**ThesisCheck** is a semantic-level thesis plagiarism detection system designed for academic scenarios. Unlike traditional string-matching plagiarism checkers, it detects academic misconduct at the **semantic understanding** level.

The system features a **Dual-Mode LLM Agent Architecture**:

- **Primary Mode: LLM Agent Mode** — Leverages large language models for deep semantic analysis, entity-relation extraction, and AI-generated content detection
- **Fallback Mode: Statistical Mode** — Pure statistical approach using Sentence-BERT + FAISS + Knowledge Graph, no API key required

Supports **two interface types**:
- **Claude Native** — Anthropic Claude via native SDK (supports token probability detection)
- **OpenAI-Compatible** — Any API that follows OpenAI Chat Completions format (DeepSeek, Qwen, GLM, etc.). Users freely configure base_url, model, and api_key

Configure any one to enable LLM Agent mode.

---

## 🎯 核心特性 | Features

| 模块 | 中文 | English |
|------|------|---------|
| **语义分析** | BERT 深度语义编码 + FAISS 向量检索，检测改写/同义替换型抄袭 | BERT-based deep semantic encoding + FAISS vector search detects paraphrasing and synonym substitution |
| **知识图谱** | 实体关系抽取 → 知识图谱构建 → 子图匹配，检测结构型抄袭 | Entity-relation extraction → knowledge graph construction → subgraph matching detects structural plagiarism |
| **AI 内容检测** | 熵、突发度、重复度、词汇丰富度、连贯性异常等多维特征分析 | Multi-dimensional analysis: entropy, burstiness, repetition, vocabulary richness, coherence anomaly |
| **LLM Agent** | 大模型驱动的深度语义对比、实体提取、AI 写作判别、报告生成 | LLM-driven deep semantic comparison, entity extraction, AI-writing detection, report generation |
| **双模式自动切换** | LLM 可用时用 Agent 模式，不可用时自动降级到统计模式 | Auto-switch between LLM Agent mode and statistical fallback |
| **学科适配** | 内置医学、计算机科学、人文学科配置，支持自定义阈值和词典 | Built-in medicine, CS, humanities configs with custom thresholds and dictionaries |
| **RESTful API** | FastAPI 异步 Web 服务，提供完整的检测和管理接口 | FastAPI async web service with complete detection and management endpoints |
| **缓存加速** | Redis 向量缓存，支持增量分析和快速重查 | Redis vector cache for incremental analysis and fast re-checking |

---

## 🤖 LLM 接口类型 | LLM Interfaces

系统提供两种接口类型，不绑定任何特定厂商，自由对接任意 API：

### 接口类型

| 类型 | 配置名 | SDK | 适用场景 | Token 概率 |
|------|--------|-----|----------|------------|
| **Claude 原生** | `claude` | Anthropic SDK | Anthropic Claude 全系列 | ✅ |
| **OpenAI 兼容** | `openai` / 任意名称 | OpenAI SDK | 任意兼容 OpenAI 格式的 API | 取决于厂商 |

> **Token 概率检测**：Claude 原生接口和部分 OpenAI 兼容接口（如 DeepSeek）支持返回 token 级 logprobs，可更准确地判断文本是否为 AI 生成。

### 配置方式

```bash
# ── 方式一：Claude 原生 ──
LLM_PROVIDER=claude
CLAUDE_API_KEY=sk-ant-xxxxxxxxxxxx
# CLAUDE_MODEL=claude-sonnet-4-6     # 可选：自定义模型

# ── 方式二：OpenAI 兼容接口（对接任意厂商） ──
# 示例：对接 DeepSeek
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-xxxxxxxxxxxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat


# 示例：对接通义千问
# LLM_PROVIDER=qwen
# LLM_API_KEY=sk-xxxxxxxxxxxx
# LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# LLM_MODEL=qwen-plus

# 示例：对接任意自定义 API
# LLM_PROVIDER=my-service
# LLM_API_KEY=xxxxxxxxxxxx
# LLM_BASE_URL=https://my-api.example.com/v1
# LLM_MODEL=my-model
```

---

## 🏗 架构总览 | Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Web Service                          │
│  POST /api/check   GET /health   GET /api/providers/disciplines     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                     PlagiarismChecker (编排器)                       │
│                                                                      │
│  ┌──────────────── LLM Agent Mode ─────────────────────────────┐   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │   │
│  │  │ SemanticAgent │  │ KGraphAgent  │  │ AIDetectionAgent │  │   │
│  │  │ 深度语义分析   │  │ 实体关系提取   │  │ AI 写作判别      │  │   │
│  │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │   │
│  │         │                 │                    │             │   │
│  │         └─────────────────┴────────────────────┘             │   │
│  │                            │                                  │   │
│  │                    ┌───────▼────────┐                        │   │
│  │                    │  LLM Clients   │                        │   │
│  │                    │  Claude / OAI  │                        │   │
│  │                    └────────────────┘                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌─────────── Statistical Fallback ──────────────────────────┐     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │     │
│  │  │SentenceEncoder│  │KnowledgeGraph│  │ AI Detection     │  │     │
│  │  │ BERT + FAISS  │  │ NetworkX     │  │ 统计特征分析      │  │     │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │     │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### LLM 客户端抽象层

```
LLMClient (抽象基类)
    ├── ClaudeClient          — 原生 Anthropic SDK
    └── OpenAICompatibleClient — 通用 OpenAI 兼容格式（对接任意 API）
         └── OpenAIClient     — OpenAI 专用（预设默认配置）
         └── 任意厂商 — 自由配置 base_url / model / api_key
```

### 模块结构

```
thesis-plagiarism-check/
├── src/thesischeck/
│   ├── core/
│   │   ├── semantic/          # BERT 编码、相似度计算、知识图谱
│   │   ├── ai_detection/      # AI 生成文本检测（特征/连贯性/指纹）
│   │   ├── config/            # 学科配置模型与动态阈值
│   │   │   └── disciplines/   # 各学科配置文件
│   │   └── report/            # 报告生成（JSON/HTML/Text）
│   ├── agents/                # LLM Agent 层
│   │   ├── base.py            # Agent 基类
│   │   ├── semantic_agent.py  # 语义对比 Agent
│   │   ├── kgraph_agent.py    # 知识图谱 Agent
│   │   ├── ai_detection_agent.py  # AI 检测 Agent
│   │   └── report_agent.py    # 报告生成 Agent
│   ├── llm/                   # LLM 客户端抽象层
│   │   ├── base.py            # LLMClient 抽象基类 + LLMConfig
│   │   ├── claude.py          # Claude 原生客户端
│   │   ├── openai_style.py    # OpenAI 兼容客户端（所有国产模型）
│   │   └── factory.py         # 工厂方法 + 自动发现
│   ├── pipeline/              # 流水线
│   │   ├── preprocessor.py    # 文本预处理
│   │   └── orchestrator.py    # 主编排器（双模式）
│   ├── api/                   # FastAPI Web 服务
│   │   └── main.py            # API 路由
│   └── cache/                 # Redis 缓存层
├── tests/                     # 测试套件（147 个测试用例）
└── pyproject.toml             # 项目配置
```

---

## 🚀 快速开始 | Quick Start

### 前置要求 | Prerequisites

- Python 3.11+
- （可选）Redis 服务器 — 启用缓存加速

### 安装 | Installation

```bash
# 克隆仓库
git clone https://github.com/LIUYUTINGNUC/thesis-plagiarism-check.git
cd thesis-plagiarism-check

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装基础依赖
pip install -e .

# 安装 LLM 支持（如需 Agent 模式）
pip install -e ".[llm]"

# 安装全部（含开发工具）
pip install -e ".[all]"
```

### 配置 LLM（可选）| Configure LLM (Optional)

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，选择接口类型和配置 API

# ── Claude 原生接口 ──
# LLM_PROVIDER=claude
# CLAUDE_API_KEY=sk-ant-xxxxxxxxxxxx

# ── OpenAI 兼容接口（对接任意厂商） ──
# LLM_PROVIDER=deepseek
# LLM_API_KEY=sk-xxxxxxxxxxxx
# LLM_BASE_URL=https://api.deepseek.com
# LLM_MODEL=deepseek-chat
```

### 启动 API 服务 | Start API Server

```bash
uvicorn src.thesischeck.api.main:app --reload --host 0.0.0.0 --port 8000
```

打开浏览器访问 `http://localhost:8000/docs` 查看交互式 API 文档。

### 命令行测试 | Test via CLI

```bash
# 使用 curl 发送查重请求
curl -X POST "http://localhost:8000/api/check" \
  -H "Content-Type: application/json" \
  -d '{
    "original_text": "深度学习在自然语言处理领域取得了显著的成果...",
    "suspect_text": "深度神经网络在自然语言处理任务中表现优异...",
    "discipline": "cs",
    "include_report": true
  }'
```

### 运行测试 | Run Tests

```bash
# 全部测试
pytest

# 仅 LLM 集成测试
pytest -m llm

# 带覆盖率
pytest --cov=src/thesischeck --cov-report=term-missing
```

---

## 📚 API 文档 | API Documentation

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查，返回 LLM 模式和版本信息 |
| `GET` | `/api/providers` | 获取所有支持的 LLM 厂商列表 |
| `GET` | `/api/disciplines` | 获取所有可用学科配置 |
| `GET` | `/api/disciplines/{name}` | 获取指定学科配置详情 |
| `POST` | `/api/check` | 执行论文查重检测（完整模式） |
| `POST` | `/api/check/simple` | 简化查重检测（查询参数） |
| `GET` | `/api/config/env` | 获取当前环境配置（不含密钥） |

启动服务后，完整交互式文档位于：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### POST /api/check 请求示例

```json
{
  "original_text": "原文内容...",
  "suspect_text": "待检测内容...",
  "discipline": "default",
  "use_llm": null,
  "include_report": false,
  "report_format": "json"
}
```

**参数说明：**
- `use_llm`: `null`=自动选择, `true`=强制LLM模式, `false`=强制统计模式
- `discipline`: 学科名称，可选 `default`, `medicine`, `cs`, `humanities`

### 响应示例

```json
{
  "overall_score": 0.85,
  "overall_verdict": "疑似抄袭",
  "semantic_similarity": 0.92,
  "kgraph_score": 0.78,
  "literal_similarity": 0.65,
  "ai_score": 0.12,
  "ai_verdict": "人类写作",
  "discipline": "cs",
  "detection_mode": "llm_agent"
}
```

---

## ⚙️ 配置学科 | Discipline Configuration

内置学科配置位于 `src/thesischeck/core/config/disciplines/`：

| 学科 | 文件名 | 适用场景 |
|------|--------|----------|
| 默认 | `default.json` | 通用配置，适用于大多数学科 |
| 医学 | `medicine.json` | 医学术语偏好，高专业术语权重 |
| 计算机科学 | `cs.json` | CS 术语偏好，代码/算法相似度检测 |
| 人文学科 | `humanities.json` | 低实体密度，高连贯性分析权重 |

每个配置文件包含相似度阈值、AI 检测权重、学科术语词典等参数。可通过 `POST /api/check` 的 `discipline` 参数选择。

---

## 🛠 技术栈 | Tech Stack

| 类别 | 技术 |
|------|------|
| **语义编码** | Sentence-BERT (`all-MiniLM-L6-v2`, 384维) |
| **向量检索** | FAISS (`IndexFlatIP`, 余弦相似度) |
| **知识图谱** | NetworkX (实体抽取 + 子图匹配) |
| **LLM 客户端** | Anthropic SDK + OpenAI SDK (多厂商兼容) |
| **Web 框架** | FastAPI + Uvicorn |
| **缓存** | Redis (可选) + 内存回退 |
| **AI 检测** | 统计特征 (熵/突发度/连贯性/词汇丰富度) |
| **报告生成** | JSON / HTML (内联CSS+SVG) / 纯文本 |
| **测试** | pytest + pytest-cov |

---

## 🤝 贡献指南 | Contributing

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

### 开发流程

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 格式化代码
ruff check src/ tests/ --fix

# 类型检查
mypy src/thesischeck/

# 运行测试
pytest --cov=src/thesischeck
```

---

## 📄 许可证 | License

[MIT License](LICENSE) © 2025 LIUYUTINGNUC

---

<div align="center">
  <sub>Built with ❤️ for academic integrity and open-source innovation</sub>
</div>