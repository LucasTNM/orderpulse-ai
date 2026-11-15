"""FastAPI entry layer exposing REST, Async SSE streaming, MCP HTTP bridge, and FinOps KPIs."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from orderpulse.config import get_settings
from orderpulse.engine import OrderPulseEngine
from orderpulse.schemas.models import AgentResolutionResponse, OrderResolveRequest


class MCPToolCallRequest(BaseModel):
    """Strict schema for HTTP-bridged MCP tool execution."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=3, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)


def _format_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def create_app(engine: OrderPulseEngine | None = None) -> FastAPI:
    """Create and configure the OrderPulse AI FastAPI application."""
    settings = get_settings()
    service_engine = engine or OrderPulseEngine(settings=settings)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Code-First Multi-Agent Decision Engine for Real-Time Delivery Triage "
            "(LangGraph, Hybrid RAG + Reranker, Redis Semantic Cache, MCP Read-Only SQL, Langfuse)."
        ),
    )
    application.state.engine = service_engine

    @application.get("/health")
    async def health_check() -> dict[str, Any]:
        eng: OrderPulseEngine = application.state.engine
        return {
            "status": "healthy",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
            "components": {
                "semantic_cache_backend": eng.cache.backend_name,
                "semantic_cache_threshold": settings.semantic_cache_threshold,
                "semantic_cache_ttl_seconds": settings.semantic_cache_ttl_seconds,
                "rag_corpus_chunks": len(eng.graph.nodes.retriever.corpus),
                "rag_pipeline": "BM25 + Dense Vector -> RRF (15) -> Cross-Encoder Rerank (3)",
                "mcp_tools_count": len(eng.graph.nodes.mcp_server.list_tools()),
                "sql_readonly_enforced": settings.sql_readonly_enforced,
                "hitl_refund_threshold_brl": settings.hitl_refund_threshold_brl,
                "langfuse_remote_connected": eng.tracer._remote_client is not None,
            },
        }

    @application.post("/v1/agent/resolve", response_model=AgentResolutionResponse)
    async def resolve_order(request: OrderResolveRequest) -> AgentResolutionResponse:
        eng: OrderPulseEngine = application.state.engine
        return await asyncio.to_thread(eng.resolve, request)

    @application.post("/v1/agent/resolve/stream")
    async def resolve_order_stream(request: OrderResolveRequest) -> StreamingResponse:
        eng: OrderPulseEngine = application.state.engine

        async def event_generator() -> AsyncGenerator[str, None]:
            captured_events: list[tuple[str, dict[str, Any]]] = []

            def _emit(event_name: str, payload: dict[str, Any]) -> None:
                captured_events.append((event_name, payload))

            response = await asyncio.to_thread(eng.resolve, request, _emit)

            for event_name, payload in captured_events:
                yield _format_sse(event_name, payload)
                await asyncio.sleep(0.005)

            # Real-time token streaming simulation over the finalized validated message
            tokens = response.final_message.split(" ")
            for idx, token in enumerate(tokens):
                chunk = token if idx == len(tokens) - 1 else f"{token} "
                yield _format_sse("token_stream", {"token": chunk, "index": idx})
                await asyncio.sleep(0.002)

            yield _format_sse("final_payload", response.model_dump())

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/v1/mcp/tools")
    async def list_mcp_tools() -> dict[str, Any]:
        eng: OrderPulseEngine = application.state.engine
        return {"tools": eng.graph.nodes.mcp_server.list_tools()}

    @application.post("/v1/mcp/call")
    async def call_mcp_tool(payload: MCPToolCallRequest) -> dict[str, Any]:
        eng: OrderPulseEngine = application.state.engine
        return await asyncio.to_thread(
            eng.graph.nodes.mcp_server.call_tool,
            payload.tool_name,
            payload.arguments,
        )

    @application.get("/v1/metrics/finops")
    async def get_finops_metrics() -> dict[str, Any]:
        eng: OrderPulseEngine = application.state.engine
        return {
            "cache_backend": eng.cache.backend_name,
            "cache_threshold": settings.semantic_cache_threshold,
            "cache_ttl_seconds": settings.semantic_cache_ttl_seconds,
            **eng.tracer.get_finops_summary(),
        }

    return application


app = create_app()
