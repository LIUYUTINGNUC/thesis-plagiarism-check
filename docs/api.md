# API 参考文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **文档**: `http://localhost:8000/docs` (Swagger UI)
- **格式**: JSON
- **编码**: UTF-8

## 端点

### 健康检查

```
GET /health
```

**响应:**
```json
{
    "status": "ok",
    "version": "0.1.0"
}
```

### 获取学科列表

```
GET /api/disciplines
```

**响应:**
```json
[
    {
        "name": "default",
        "display_name": "通用默认配置",
        "description": "适用于大多数学科的通用查重配置"
    },
    {
        "name": "medicine",
        "display_name": "医学",
        "description": "医学论文查重配置：允许较高引用率，强调新发现和创新性"
    },
    {
        "name": "cs",
        "display_name": "计算机科学",
        "description": "计算机科学论文查重配置：中等引用率，强调方法论贡献"
    },
    {
        "name": "humanities",
        "display_name": "人文社科",
        "description": "人文社科论文查重配置：低引用容忍度，强调论点原创性"
    }
]
```

### 获取学科配置详情

```
GET /api/disciplines/{name}
```

**参数:**
- `name`: 学科名称（如 `medicine`, `cs`, `humanities`）

**响应:** 完整学科配置对象。

### 执行查重检测

```
POST /api/check
```

**请求体:**
```json
{
    "original_text": "原始论文的完整文本内容...",
    "suspect_text": "待检测论文的完整文本内容...",
    "discipline": "default",
    "include_report": false,
    "report_format": "json"
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| original_text | string | — | 原始论文文本（至少10字符） |
| suspect_text | string | — | 待检测文本（至少10字符） |
| discipline | string | "default" | 学科配置名称 |
| include_report | bool | false | 是否在响应中包含完整报告 |
| report_format | string | "json" | 报告格式：json/text/html |

**响应:**
```json
{
    "overall_score": 0.2345,
    "overall_verdict": "distinct",
    "semantic_similarity": 0.1234,
    "kgraph_score": 0.3456,
    "literal_similarity": 0.1011,
    "ai_score": 0.4321,
    "ai_verdict": "likely_human_written",
    "discipline": "default",
    "report": null
}
```

**判定结论说明:**

| verdict | 含义 |
|---------|------|
| highly_similar | 高度相似 — 建议详细审查 |
| moderately_similar | 中度相似 — 需要进一步检查 |
| slightly_similar | 轻度相似 — 可参考 |
| distinct | 不相似 — 通过检测 |

**AI检测判定说明:**

| ai_verdict | 含义 |
|------------|------|
| likely_ai_generated | 极可能为AI生成 |
| possibly_ai_generated | 可能为AI生成 |
| possibly_human_written | 可能为人类写作 |
| likely_human_written | 极可能为人类写作 |

### 简化查重接口

```
POST /api/check/simple?original_text=...&suspect_text=...&discipline=default
```

适用于快速测试的表单参数接口。

## 错误码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 404 | 学科配置不存在 |
| 422 | 请求参数验证失败 |
| 500 | 服务器内部错误 |

## 启动服务

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API
uvicorn src.thesischeck.api.main:app --reload --host 0.0.0.0 --port 8000

# 或在项目根目录
cd thesis-plagiarism-check
python -m src.thesischeck.api.main
```