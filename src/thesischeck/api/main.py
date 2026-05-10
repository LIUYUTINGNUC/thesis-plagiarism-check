"""FastAPI 应用——提供论文查重检测的 REST API。

支持双模式：
- LLM Agent 模式（设置环境变量 LLM_PROVIDER+API_KEY 后自动启用）
- 统计模式（默认，无需 API key）
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from thesischeck.core.config.loader import list_available_disciplines, load_discipline_config
from thesischeck.core.report.generator import ReportGenerator
from thesischeck.pipeline.orchestrator import PlagiarismChecker

app = FastAPI(
    title="语义级论文查重检测系统 API",
    description="基于 BERT 语义分析、知识图谱、AI 内容检测和 LLM Agent 的论文查重服务。"
                "支持 Claude / OpenAI / DeepSeek / 通义千问 / 智谱GLM 等多种大模型。",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# 数据模型
# ======================================================================


class CheckRequest(BaseModel):
    """查重检测请求。"""
    original_text: str = Field(..., min_length=10, description="原始论文文本")
    suspect_text: str = Field(..., min_length=10, description="待检测论文文本")
    discipline: str = Field(default="default", description="学科名称")
    use_llm: Optional[bool] = Field(
        default=None,
        description="是否使用 LLM Agent 模式。null=自动, true=强制LLM, false=强制统计",
    )
    include_report: bool = Field(default=False, description="响应中包含完整报告")
    report_format: str = Field(default="json", description="报告格式: json / text / html")


class CheckResponse(BaseModel):
    """查重检测响应。"""
    overall_score: float
    overall_verdict: str
    semantic_similarity: float
    kgraph_score: float
    literal_similarity: float
    ai_score: float
    ai_verdict: str
    discipline: str
    detection_mode: str = Field(default="statistical", description="检测模式: llm_agent / statistical")
    report: Optional[str] = None


class DisciplineInfo(BaseModel):
    """学科配置信息。"""
    name: str
    display_name: str
    description: str


class ProviderInfo(BaseModel):
    """LLM 厂商信息。"""
    name: str
    description: str


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str = "ok"
    version: str = "0.2.0"
    llm_provider: Optional[str] = None
    llm_available: bool = False
    detection_mode: str = "statistical"


# ======================================================================
# 全局 checker 实例（惰性初始化，复用 LLM 客户端）
# ======================================================================

_checker_instance: Optional[PlagiarismChecker] = None


def _get_checker(discipline: str = "default") -> PlagiarismChecker:
    """获取（或创建）全局 checker 实例。

    复用 LLM 客户端以避免重复初始化。
    """
    global _checker_instance
    if _checker_instance is None or _checker_instance.discipline_name != discipline:
        _checker_instance = PlagiarismChecker(discipline=discipline)
    return _checker_instance


# ======================================================================
# API 路由
# ======================================================================


@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查端点。返回系统状态和 LLM 模式信息。"""
    if _checker_instance is None:
        checker = _get_checker()
    else:
        checker = _checker_instance

    return HealthResponse(
        llm_provider=os.getenv("LLM_PROVIDER"),
        llm_available=checker.llm_mode,
        detection_mode="llm_agent" if checker.llm_mode else "statistical",
    )


@app.get("/api/providers", response_model=list[ProviderInfo], tags=["LLM配置"])
async def get_llm_providers():
    """获取所有支持的 LLM 厂商列表。"""
    try:
        from thesischeck.llm.factory import list_available_providers
        return [ProviderInfo(**p) for p in list_available_providers()]
    except ImportError:
        return [
            ProviderInfo(name="claude", description="Anthropic Claude"),
            ProviderInfo(name="openai", description="OpenAI GPT"),
        ]


@app.get("/api/disciplines", response_model=list[DisciplineInfo], tags=["学科配置"])
async def get_disciplines():
    """获取所有可用学科配置列表。"""
    return [DisciplineInfo(**d) for d in list_available_disciplines()]


@app.get("/api/disciplines/{name}", response_model=dict, tags=["学科配置"])
async def get_discipline_config(name: str):
    """获取指定学科的配置详情。"""
    try:
        config = load_discipline_config(name)
        return config.model_dump()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"不存在的学科配置: {name}")


@app.post("/api/check", response_model=CheckResponse, tags=["查重检测"])
async def run_check(request: CheckRequest):
    """执行论文查重检测。

    支持双模式：
    1. LLM Agent 模式：设置环境变量 LLM_PROVIDER 和对应 API_KEY 后自动启用
    2. 统计模式：默认模式，无需任何 API key

    当 LLM 不可用时自动降级到统计模式。
    """
    try:
        checker = _get_checker(discipline=request.discipline)
        result = checker.check(
            original_text=request.original_text,
            suspect_text=request.suspect_text,
            use_llm=request.use_llm,
        )

        mode = result.details.get("mode", "statistical")
        response = CheckResponse(
            overall_score=result.overall_score,
            overall_verdict=result.overall_verdict,
            semantic_similarity=result.semantic_similarity,
            kgraph_score=result.kgraph_score,
            literal_similarity=result.literal_similarity,
            ai_score=result.ai_score,
            ai_verdict=result.ai_verdict,
            discipline=result.discipline,
            detection_mode=mode,
        )

        if request.include_report:
            generator = ReportGenerator()
            if request.report_format == "html":
                response.report = generator.generate_html_report(result)
            elif request.report_format == "text":
                response.report = generator.generate_text_report(result)
            else:
                response.report = generator.generate_json_report(result)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测过程出错: {str(e)}")


@app.post("/api/check/simple", response_model=dict, tags=["查重检测"])
async def run_simple_check(
    original_text: str = Query(..., min_length=10),
    suspect_text: str = Query(..., min_length=10),
    discipline: str = "default",
):
    """简化的查重检测接口（查询参数）。"""
    checker = _get_checker(discipline=discipline)
    result = checker.check(original_text=original_text, suspect_text=suspect_text)
    return {
        "overall_score": result.overall_score,
        "overall_verdict": result.overall_verdict,
        "semantic_similarity": result.semantic_similarity,
        "ai_score": result.ai_score,
        "discipline": result.discipline,
        "detection_mode": result.details.get("mode", "statistical"),
    }


@app.get("/api/config/env", response_model=dict, tags=["系统"])
async def get_env_config():
    """获取当前环境配置（不含密钥明文）。"""
    provider = os.getenv("LLM_PROVIDER", "")
    has_key = bool(os.getenv(f"{provider.upper()}_API_KEY") or os.getenv("LLM_API_KEY"))
    return {
        "llm_provider": provider or "未配置（使用统计模式）",
        "llm_configured": has_key,
        "detection_mode": "llm_agent" if has_key else "statistical",
        "note": "LLM 模式下 API 密钥不会在响应中暴露",
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)