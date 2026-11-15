"""Cyclic Multi-Agent State Machine (LangGraph-compatible StateGraph).

Graph topology:
  START -> triage_guardrail_node
    |-- (BLOCKED / OUT_OF_SCOPE) -> END
    |-- (SAFE & IN_SCOPE) -> policy_agent_node -> operational_agent_node -> judge_validator_node
                                    ^                                               |
                                    |-------- (revision_requested == True) ---------|
                                    |-------- (approved == True) -----------------> END
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from orderpulse.agents.nodes import EventCallback, OrderPulseNodes
from orderpulse.config import Settings, get_settings
from orderpulse.schemas.models import (
    AgentGraphState,
    GuardrailStatus,
    IntentCategory,
    OrderResolveRequest,
)


class _LangGraphPayload(TypedDict):
    state: AgentGraphState


def route_after_triage(state: AgentGraphState) -> str:
    """Conditional edge: short-circuit blocked or out-of-scope requests."""
    triage = state.triage
    if triage is None:
        return "end"
    if triage.guardrail_status != GuardrailStatus.SAFE:
        return "end"
    if triage.intent == IntentCategory.OUT_OF_SCOPE:
        return "end"
    return "policy_agent_node"


def route_after_judge(state: AgentGraphState) -> str:
    """Conditional edge: trigger cyclic Self-Reflection revision or finalize."""
    verdict = state.judge_verdict
    if verdict is not None and verdict.revision_requested:
        return "operational_agent_node"
    return "end"


class OrderPulseStateGraph:
    """Orchestrates the 4-node cyclic multi-agent decision graph."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        nodes: Optional[OrderPulseNodes] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.nodes = nodes or OrderPulseNodes(settings=self.settings)
        self._compiled_langgraph = self._try_build_langgraph()

    def _try_build_langgraph(self) -> Any:
        """Build a native ``langgraph.graph.StateGraph`` when langgraph is installed."""
        try:
            from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]
        except ImportError:
            return None

        builder = StateGraph(_LangGraphPayload)
        builder.add_node(
            "triage_guardrail_node",
            lambda p: {"state": self.nodes.triage_guardrail_node(p["state"])},
        )
        builder.add_node(
            "policy_agent_node",
            lambda p: {"state": self.nodes.policy_agent_node(p["state"])},
        )
        builder.add_node(
            "operational_agent_node",
            lambda p: {"state": self.nodes.operational_agent_node(p["state"])},
        )
        builder.add_node(
            "judge_validator_node",
            lambda p: {"state": self.nodes.judge_validator_node(p["state"])},
        )

        builder.add_edge(START, "triage_guardrail_node")
        builder.add_conditional_edges(
            "triage_guardrail_node",
            lambda p: route_after_triage(p["state"]),
            {"policy_agent_node": "policy_agent_node", "end": END},
        )
        builder.add_edge("policy_agent_node", "operational_agent_node")
        builder.add_edge("operational_agent_node", "judge_validator_node")
        builder.add_conditional_edges(
            "judge_validator_node",
            lambda p: route_after_judge(p["state"]),
            {"operational_agent_node": "operational_agent_node", "end": END},
        )
        return builder.compile()

    def run(
        self,
        request: OrderResolveRequest,
        trace_id: str,
        emit: EventCallback = None,
    ) -> AgentGraphState:
        """Execute the cyclic state machine from START to END."""
        state = AgentGraphState(trace_id=trace_id, request=request)

        # When streaming events are not requested and LangGraph is installed, run compiled graph.
        if emit is None and self._compiled_langgraph is not None:
            result = self._compiled_langgraph.invoke({"state": state})
            return result["state"]

        # Deterministic cyclic state machine execution (identical topology & conditional edges).
        state = self.nodes.triage_guardrail_node(state, emit=emit)
        if route_after_triage(state) == "end":
            return state

        state = self.nodes.policy_agent_node(state, emit=emit)

        while True:
            state = self.nodes.operational_agent_node(state, emit=emit)
            state = self.nodes.judge_validator_node(state, emit=emit)
            if route_after_judge(state) == "end":
                break

        return state
