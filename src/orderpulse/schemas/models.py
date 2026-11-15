"""Strict Pydantic v2 data models and LangGraph state contracts.

All domain models enforce ``extra="forbid"`` to prevent silent schema drift
between LLM structured outputs, MCP tools, and API responses.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class IntentCategory(str, Enum):
    """Supported routing intents for logistics and order resolution."""

    DELIVERY_DELAY_DISPUTE = "DELIVERY_DELAY_DISPUTE"
    REFUND_REQUEST = "REFUND_REQUEST"
    ORDER_CANCELLATION = "ORDER_CANCELLATION"
    CATALOG_OR_ITEM_ISSUE = "CATALOG_OR_ITEM_ISSUE"
    FRAUD_OR_SECURITY_ALERT = "FRAUD_OR_SECURITY_ALERT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class GuardrailStatus(str, Enum):
    """Input security & policy guardrail verdicts."""

    SAFE = "SAFE"
    PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
    PII_LEAK_RISK = "PII_LEAK_RISK"
    POLICY_VIOLATION = "POLICY_VIOLATION"


class ResolutionDecision(str, Enum):
    """Final autonomous decision outcomes enforced by the Judge node."""

    AUTO_APPROVED_REFUND = "AUTO_APPROVED_REFUND"
    PARTIAL_VOUCHER = "PARTIAL_VOUCHER"
    ESCALATED_HITL = "ESCALATED_HITL"
    REJECTED_POLICY = "REJECTED_POLICY"
    BLOCKED_GUARDRAIL = "BLOCKED_GUARDRAIL"
    INFO_PROVIDED = "INFO_PROVIDED"


class OrderResolveRequest(BaseModel):
    """Inbound payload for POST /v1/agent/resolve."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(..., min_length=3, max_length=2000, description="User complaint or inquiry.")
    order_id: Optional[str] = Field(
        default=None,
        pattern=r"^ORD-\d{4,8}$",
        description="Optional order identifier (e.g., ORD-9001).",
    )
    customer_id: Optional[str] = Field(default=None, max_length=64)
    channel: str = Field(default="api", max_length=32)
    bypass_cache: bool = Field(
        default=False,
        description="Force full multi-agent graph execution bypassing semantic cache.",
    )


class TriageClassification(BaseModel):
    """Strict structured output returned by the Intent Router & Guardrail."""

    model_config = ConfigDict(extra="forbid")

    intent: IntentCategory
    guardrail_status: GuardrailStatus
    guardrail_reason: str = Field(default="Passed security and safety checks.")
    extracted_order_id: Optional[str] = Field(default=None)
    claimed_amount_brl: Optional[float] = Field(default=None, ge=0.0)
    urgency_score: int = Field(..., ge=1, le=5)
    requires_sql_lookup: bool
    requires_policy_rag: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class PolicyChunk(BaseModel):
    """Retrieved business policy document with hybrid & reranker scoring metadata."""

    model_config = ConfigDict(extra="forbid")

    policy_id: str
    title: str
    category: str
    content: str
    bm25_score: float = Field(default=0.0)
    dense_score: float = Field(default=0.0)
    hybrid_rrf_score: float = Field(default=0.0)
    rerank_score: float = Field(default=0.0)


class OrderOperationalRecord(BaseModel):
    """Read-only snapshot of an order fetched via the MCP SQL server."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    customer_id: str
    customer_tier: str
    restaurant_name: str
    order_status: str
    total_amount_brl: float = Field(..., ge=0.0)
    promised_eta_min: int = Field(..., ge=1)
    actual_delivery_min: Optional[int] = Field(default=None, ge=0)
    delay_minutes: int = Field(default=0, ge=0)
    missing_items_reported: bool = Field(default=False)
    missing_items_value_brl: float = Field(default=0.0, ge=0.0)
    prior_refunds_30d: int = Field(default=0, ge=0)
    created_at_iso: str
    courier_gps_delayed: bool = Field(default=False)


class JudgeVerdict(BaseModel):
    """Self-reflection validation result verifying financial and policy compliance."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    final_decision: ResolutionDecision
    approved_refund_brl: float = Field(default=0.0, ge=0.0)
    requires_human_in_the_loop: bool = Field(default=False)
    revision_requested: bool = Field(default=False)
    reflection_notes: list[str] = Field(default_factory=list)
    policy_citations: list[str] = Field(default_factory=list)


class NodeTraceMetric(BaseModel):
    """Per-node LLMOps latency, token usage, and USD cost metric."""

    model_config = ConfigDict(extra="forbid")

    node_name: str
    latency_ms: float = Field(..., ge=0.0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class AgentResolutionResponse(BaseModel):
    """Final contract returned to API callers and evaluated by RAGAS."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    cache_hit: bool = Field(default=False)
    cache_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    decision: ResolutionDecision
    requires_human_in_the_loop: bool = Field(default=False)
    approved_refund_brl: float = Field(default=0.0, ge=0.0)
    triage: TriageClassification
    retrieved_policies: list[PolicyChunk] = Field(default_factory=list)
    operational_record: Optional[OrderOperationalRecord] = Field(default=None)
    judge_verdict: JudgeVerdict
    final_message: str
    total_latency_ms: float = Field(..., ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    cost_saved_usd: float = Field(default=0.0, ge=0.0)
    node_metrics: list[NodeTraceMetric] = Field(default_factory=list)


class AgentGraphState(BaseModel):
    """Mutable state carried across cyclic LangGraph node transitions."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    request: OrderResolveRequest
    triage: Optional[TriageClassification] = None
    retrieved_policies: list[PolicyChunk] = Field(default_factory=list)
    initial_candidates_count: int = Field(default=0, ge=0)
    operational_record: Optional[OrderOperationalRecord] = None
    draft_decision: Optional[ResolutionDecision] = None
    draft_refund_brl: float = Field(default=0.0, ge=0.0)
    draft_message: str = Field(default="")
    draft_citations: list[str] = Field(default_factory=list)
    judge_verdict: Optional[JudgeVerdict] = None
    revision_count: int = Field(default=0, ge=0)
    node_metrics: list[NodeTraceMetric] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
