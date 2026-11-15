"""Langfuse-compatible ``@observe()`` instrumentation and FinOps trace aggregator.

Captures per-node latencies, prompt/completion tokens, USD costs, and Semantic
Cache savings in-process, and optionally ships spans to a live Langfuse server
when ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are configured.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

from orderpulse.config import Settings, get_settings
from orderpulse.schemas.models import AgentResolutionResponse

F = TypeVar("F", bound=Callable[..., Any])

_ACTIVE_SPANS: contextvars.ContextVar[Optional[list[dict[str, Any]]]] = contextvars.ContextVar(
    "orderpulse_active_spans", default=None
)


def observe(name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator compatible with Langfuse ``@observe()`` for sync and async functions."""

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    spans = _ACTIVE_SPANS.get()
                    if spans is not None:
                        spans.append(
                            {
                                "span_id": uuid.uuid4().hex[:12],
                                "name": span_name,
                                "latency_ms": elapsed_ms,
                            }
                        )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                spans = _ACTIVE_SPANS.get()
                if spans is not None:
                    spans.append(
                        {
                            "span_id": uuid.uuid4().hex[:12],
                            "name": span_name,
                            "latency_ms": elapsed_ms,
                        }
                    )

        return sync_wrapper  # type: ignore[return-value]

    return decorator


class LangfuseTracer:
    """Stores execution traces and computes FinOps latency/cost KPIs."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.traces: list[dict[str, Any]] = []
        self._remote_client = self._init_remote_langfuse()

    def _init_remote_langfuse(self) -> Any:
        if not (self.settings.langfuse_public_key and self.settings.langfuse_secret_key):
            return None
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found]

            return Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
        except Exception:
            return None

    @contextmanager
    def capture_spans(self) -> Iterator[list[dict[str, Any]]]:
        """Context manager collecting @observe() span timings for a request."""
        collected: list[dict[str, Any]] = []
        token = _ACTIVE_SPANS.set(collected)
        try:
            yield collected
        finally:
            _ACTIVE_SPANS.reset(token)

    def record_resolution(
        self,
        response: AgentResolutionResponse,
        query: str,
        spans: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Record a full agent execution or semantic cache hit trace."""
        prompt_tokens = sum(m.prompt_tokens for m in response.node_metrics)
        completion_tokens = sum(m.completion_tokens for m in response.node_metrics)

        trace_entry = {
            "trace_id": response.trace_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "intent": response.triage.intent.value,
            "decision": response.decision.value,
            "cache_hit": response.cache_hit,
            "cache_similarity": response.cache_similarity,
            "requires_human_in_the_loop": response.requires_human_in_the_loop,
            "total_latency_ms": response.total_latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "total_cost_usd": response.total_cost_usd,
            "cost_saved_usd": response.cost_saved_usd,
            "node_metrics": [m.model_dump() for m in response.node_metrics],
            "spans": spans or [],
        }
        self.traces.append(trace_entry)

        if self._remote_client is not None:
            try:
                lf_trace = self._remote_client.trace(
                    id=response.trace_id,
                    name="orderpulse_agent_resolve",
                    input={"query": query},
                    output={
                        "decision": response.decision.value,
                        "cache_hit": response.cache_hit,
                        "approved_refund_brl": response.approved_refund_brl,
                    },
                    metadata={
                        "cache_hit": response.cache_hit,
                        "cost_usd": response.total_cost_usd,
                        "cost_saved_usd": response.cost_saved_usd,
                    },
                )
                for metric in response.node_metrics:
                    lf_trace.span(
                        name=metric.node_name,
                        metadata={
                            "latency_ms": metric.latency_ms,
                            "prompt_tokens": metric.prompt_tokens,
                            "completion_tokens": metric.completion_tokens,
                            "cost_usd": metric.cost_usd,
                        },
                    )
            except Exception:
                pass

        return trace_entry

    def get_finops_summary(self) -> dict[str, Any]:
        """Aggregate FinOps metrics comparing Cold Graph runs vs Semantic Cache hits."""
        total = len(self.traces)
        hits = [t for t in self.traces if t["cache_hit"]]
        misses = [t for t in self.traces if not t["cache_hit"]]

        avg_cold_ms = (
            round(sum(t["total_latency_ms"] for t in misses) / len(misses), 3)
            if misses
            else 0.0
        )
        avg_hit_ms = (
            round(sum(t["total_latency_ms"] for t in hits) / len(hits), 3)
            if hits
            else 0.0
        )
        latency_reduction = (
            round(((avg_cold_ms - avg_hit_ms) / avg_cold_ms) * 100.0, 2)
            if avg_cold_ms > 0 and hits
            else 0.0
        )

        return {
            "total_requests": total,
            "cache_hits": len(hits),
            "cache_misses": len(misses),
            "cache_hit_rate": round(len(hits) / total, 4) if total else 0.0,
            "avg_cold_latency_ms": avg_cold_ms,
            "avg_cache_hit_latency_ms": avg_hit_ms,
            "latency_reduction_pct": latency_reduction,
            "total_tokens_consumed": sum(t["total_tokens"] for t in self.traces),
            "total_cost_usd": round(sum(t["total_cost_usd"] for t in self.traces), 8),
            "total_cost_saved_usd": round(sum(t["cost_saved_usd"] for t in self.traces), 8),
            "langfuse_remote_connected": self._remote_client is not None,
        }

    def export_summary(self, output_path: str | Path = "reports/langfuse_traces_summary.json") -> Path:
        """Persist FinOps KPIs and recent traces to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "finops_summary": self.get_finops_summary(),
            "recent_traces": self.traces[-25:],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def reset(self) -> None:
        """Clear collected in-memory traces."""
        self.traces.clear()


_DEFAULT_TRACER: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    """Return process-wide LangfuseTracer singleton."""
    global _DEFAULT_TRACER
    if _DEFAULT_TRACER is None:
        _DEFAULT_TRACER = LangfuseTracer()
    return _DEFAULT_TRACER
