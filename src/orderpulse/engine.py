"""End-to-end OrderPulse AI Engine integrating Semantic Cache, LangGraph, and Langfuse."""

from __future__ import annotations

import time
import uuid
from typing import Optional

from orderpulse.agents.graph import OrderPulseStateGraph
from orderpulse.agents.nodes import EventCallback
from orderpulse.cache.semantic_cache import RedisSemanticCache
from orderpulse.config import Settings, get_settings
from orderpulse.observability.langfuse_tracer import LangfuseTracer, get_tracer, observe
from orderpulse.schemas.models import (
    AgentResolutionResponse,
    JudgeVerdict,
    OrderResolveRequest,
    ResolutionDecision,
    TriageClassification,
    GuardrailStatus,
    IntentCategory,
)


class OrderPulseEngine:
    """High-level facade coordinating Semantic Cache -> LangGraph -> Langfuse."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache: Optional[RedisSemanticCache] = None,
        graph: Optional[OrderPulseStateGraph] = None,
        tracer: Optional[LangfuseTracer] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or RedisSemanticCache(settings=self.settings)
        self.graph = graph or OrderPulseStateGraph(settings=self.settings)
        self.tracer = tracer or get_tracer()

    @observe(name="orderpulse_resolve")
    def resolve(
        self,
        request: OrderResolveRequest,
        emit: EventCallback = None,
    ) -> AgentResolutionResponse:
        """Resolve an order/policy inquiry with Semantic Cache shortcut and full tracing."""
        t0 = time.perf_counter()
        trace_id = f"tr-{uuid.uuid4().hex[:16]}"

        with self.tracer.capture_spans() as spans:
            if not request.bypass_cache:
                cache_res = self.cache.lookup(query=request.query, order_id=request.order_id)
                if emit:
                    emit(
                        "cache_check",
                        {
                            "cache_hit": cache_res.hit,
                            "similarity": cache_res.similarity,
                            "backend": self.cache.backend_name,
                        },
                    )
                if cache_res.hit and cache_res.response is not None:
                    elapsed_ms = max(0.05, round((time.perf_counter() - t0) * 1000.0, 3))
                    cached = cache_res.response.model_copy(
                        update={
                            "trace_id": trace_id,
                            "cache_hit": True,
                            "cache_similarity": cache_res.similarity,
                            "total_latency_ms": elapsed_ms,
                            "total_cost_usd": 0.0,
                            "cost_saved_usd": cache_res.original_cost_usd,
                            "node_metrics": [],
                        }
                    )
                    self.tracer.record_resolution(cached, query=request.query, spans=spans)
                    return cached
            elif emit:
                emit(
                    "cache_check",
                    {
                        "cache_hit": False,
                        "similarity": None,
                        "bypassed": True,
                    },
                )

            final_state = self.graph.run(request=request, trace_id=trace_id, emit=emit)
            total_ms = max(0.1, round((time.perf_counter() - t0) * 1000.0, 3))
            total_cost = round(sum(m.cost_usd for m in final_state.node_metrics), 8)

            triage = final_state.triage or TriageClassification(
                intent=IntentCategory.OUT_OF_SCOPE,
                guardrail_status=GuardrailStatus.SAFE,
                urgency_score=1,
                requires_sql_lookup=False,
                requires_policy_rag=False,
                confidence=0.5,
            )
            verdict = final_state.judge_verdict or JudgeVerdict(
                approved=True,
                final_decision=final_state.draft_decision or ResolutionDecision.INFO_PROVIDED,
                approved_refund_brl=final_state.draft_refund_brl,
                requires_human_in_the_loop=False,
                revision_requested=False,
                reflection_notes=[],
                policy_citations=final_state.draft_citations,
            )

            response = AgentResolutionResponse(
                trace_id=trace_id,
                cache_hit=False,
                cache_similarity=None,
                decision=verdict.final_decision,
                requires_human_in_the_loop=verdict.requires_human_in_the_loop,
                approved_refund_brl=verdict.approved_refund_brl,
                triage=triage,
                retrieved_policies=final_state.retrieved_policies,
                operational_record=final_state.operational_record,
                judge_verdict=verdict,
                final_message=final_state.draft_message,
                total_latency_ms=total_ms,
                total_cost_usd=total_cost,
                cost_saved_usd=0.0,
                node_metrics=final_state.node_metrics,
            )

            self.cache.store(query=request.query, response=response, order_id=request.order_id)
            self.tracer.record_resolution(response, query=request.query, spans=spans)
            return response
