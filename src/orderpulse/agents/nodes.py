"""LangGraph execution nodes: Triage/Guardrail, Policy RAG, Operational MCP, and Judge."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from orderpulse.agents.router_guardrail import IntentRouterGuardrail
from orderpulse.config import Settings, get_settings
from orderpulse.mcp.server import OrderPulseMCPServer
from orderpulse.observability.langfuse_tracer import observe
from orderpulse.rag.hybrid_retriever import HybridPolicyRetriever
from orderpulse.schemas.models import (
    AgentGraphState,
    GuardrailStatus,
    IntentCategory,
    JudgeVerdict,
    NodeTraceMetric,
    OrderOperationalRecord,
    ResolutionDecision,
)

EventCallback = Optional[Callable[[str, dict[str, Any]], None]]


def _estimate_cost(prompt_tokens: int, completion_tokens: int, is_router: bool = False) -> float:
    """Compute realistic API cost in USD (gpt-4o-mini / llama-3.3-70b pricing)."""
    in_rate = 0.00000015 if is_router else 0.00000030
    out_rate = 0.00000060 if is_router else 0.00000120
    return round((prompt_tokens * in_rate) + (completion_tokens * out_rate), 8)


class OrderPulseNodes:
    """Container holding stateful adapters used across LangGraph nodes."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        router: Optional[IntentRouterGuardrail] = None,
        retriever: Optional[HybridPolicyRetriever] = None,
        mcp_server: Optional[OrderPulseMCPServer] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.router = router or IntentRouterGuardrail(self.settings)
        self.retriever = retriever or HybridPolicyRetriever(settings=self.settings)
        self.mcp_server = mcp_server or OrderPulseMCPServer()

    @observe(name="triage_guardrail_node")
    def triage_guardrail_node(
        self,
        state: AgentGraphState,
        emit: EventCallback = None,
    ) -> AgentGraphState:
        """Node 0: Run input security guardrails and classify intent into strict schema."""
        t0 = time.perf_counter()
        triage = self.router.classify(state.request)
        state.triage = triage

        prompt_tok = max(40, len(state.request.query.split()) * 3)
        comp_tok = 48
        latency_ms = max(0.05, round((time.perf_counter() - t0) * 1000.0, 3))
        state.node_metrics.append(
            NodeTraceMetric(
                node_name="triage_guardrail_node",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                cost_usd=_estimate_cost(prompt_tok, comp_tok, is_router=True),
            )
        )

        if triage.guardrail_status != GuardrailStatus.SAFE:
            state.draft_decision = ResolutionDecision.BLOCKED_GUARDRAIL
            state.draft_refund_brl = 0.0
            state.draft_citations = ["POL-SEC-018"]
            state.draft_message = (
                f"Solicitacao bloqueada pelo Guardrail de Seguranca ({triage.guardrail_status.value}): "
                f"{triage.guardrail_reason} Nenhuma operacao ou reembolso foi processado (POL-SEC-018)."
            )
            state.judge_verdict = JudgeVerdict(
                approved=True,
                final_decision=ResolutionDecision.BLOCKED_GUARDRAIL,
                approved_refund_brl=0.0,
                requires_human_in_the_loop=False,
                revision_requested=False,
                reflection_notes=[triage.guardrail_reason],
                policy_citations=["POL-SEC-018"],
            )
        elif triage.intent == IntentCategory.OUT_OF_SCOPE:
            state.draft_decision = ResolutionDecision.INFO_PROVIDED
            state.draft_refund_brl = 0.0
            state.draft_citations = []
            state.draft_message = (
                "Sou o agente especializado em logistica, entregas e pedidos do OrderPulse AI. "
                "Posso ajudar com atrasos, itens faltantes, cancelamentos ou duvidas de reembolso "
                "informando o codigo do pedido (ex.: ORD-9001)."
            )
            state.judge_verdict = JudgeVerdict(
                approved=True,
                final_decision=ResolutionDecision.INFO_PROVIDED,
                approved_refund_brl=0.0,
                requires_human_in_the_loop=False,
                revision_requested=False,
                reflection_notes=["Out-of-scope inquiry handled gracefully without downstream RAG/SQL cost."],
                policy_citations=[],
            )

        if emit:
            emit(
                "triage_completed",
                {
                    "intent": triage.intent.value,
                    "guardrail_status": triage.guardrail_status.value,
                    "extracted_order_id": triage.extracted_order_id,
                    "urgency_score": triage.urgency_score,
                },
            )
        return state

    @observe(name="policy_agent_node")
    def policy_agent_node(
        self,
        state: AgentGraphState,
        emit: EventCallback = None,
    ) -> AgentGraphState:
        """Node 1: Execute Hybrid RAG (BM25 + Dense + RRF top 15 -> Reranker top 3)."""
        t0 = time.perf_counter()
        triage = state.triage
        enriched_query = state.request.query
        if triage is not None:
            hints: list[str] = []
            if triage.claimed_amount_brl and triage.claimed_amount_brl > self.settings.hitl_refund_threshold_brl:
                hints.append("POL-FIN-150 alcada financeira 150 aprovacao humana HITL")
            if hints:
                enriched_query = f"{state.request.query} {' '.join(hints)}"

        top_chunks, initial_count = self.retriever.retrieve(enriched_query)
        state.retrieved_policies = top_chunks
        state.initial_candidates_count = initial_count

        prompt_tok = 180 + sum(len(c.content.split()) for c in top_chunks)
        comp_tok = 65
        latency_ms = max(0.05, round((time.perf_counter() - t0) * 1000.0, 3))
        state.node_metrics.append(
            NodeTraceMetric(
                node_name="policy_agent_node",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                cost_usd=_estimate_cost(prompt_tok, comp_tok),
            )
        )

        if emit:
            emit(
                "rag_reranked",
                {
                    "initial_candidates": initial_count,
                    "reranked_top_k": len(top_chunks),
                    "policy_ids": [c.policy_id for c in top_chunks],
                },
            )
        return state

    @observe(name="operational_agent_node")
    def operational_agent_node(
        self,
        state: AgentGraphState,
        emit: EventCallback = None,
    ) -> AgentGraphState:
        """Node 2: Query Read-Only SQL via MCP tools and synthesize (or revise) decision."""
        t0 = time.perf_counter()
        triage = state.triage
        order_id = triage.extracted_order_id if triage else state.request.order_id

        record = state.operational_record
        if record is None and order_id:
            order_resp = self.mcp_server.call_tool("get_order_status", {"order_id": order_id})
            if not order_resp.get("isError") and order_resp.get("found"):
                record = OrderOperationalRecord.model_validate(order_resp["record"])
                state.operational_record = record
                self.mcp_server.call_tool(
                    "get_customer_refund_history",
                    {"customer_id": record.customer_id},
                )

        # Handle Self-Reflection revision loop requested by the Judge node
        if state.judge_verdict is not None and state.judge_verdict.revision_requested:
            self._apply_judge_revision(state)
        else:
            self._draft_initial_proposal(state)

        prompt_tok = 260
        comp_tok = 110
        latency_ms = max(0.05, round((time.perf_counter() - t0) * 1000.0, 3))
        state.node_metrics.append(
            NodeTraceMetric(
                node_name="operational_agent_node",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                cost_usd=_estimate_cost(prompt_tok, comp_tok),
            )
        )

        if emit:
            emit(
                "mcp_sql_queried",
                {
                    "order_id": record.order_id if record else None,
                    "order_status": record.order_status if record else None,
                    "draft_decision": state.draft_decision.value if state.draft_decision else None,
                    "draft_refund_brl": state.draft_refund_brl,
                    "revision_count": state.revision_count,
                },
            )
        return state

    @observe(name="judge_validator_node")
    def judge_validator_node(
        self,
        state: AgentGraphState,
        emit: EventCallback = None,
    ) -> AgentGraphState:
        """Node 3: Self-Reflection validator enforcing R$ 150 HITL, order cap, and abuse rules."""
        t0 = time.perf_counter()
        threshold = self.settings.hitl_refund_threshold_brl
        abuse_limit = self.settings.max_refunds_30d_abuse_threshold
        record = state.operational_record
        decision = state.draft_decision or ResolutionDecision.INFO_PROVIDED
        refund = round(state.draft_refund_brl, 2)
        citations = list(dict.fromkeys(state.draft_citations))
        notes: list[str] = []
        revision_needed = False

        if record is not None:
            # Rule 1: Cap refund to total order amount (POL-ABU-010)
            if refund > record.total_amount_brl:
                revision_needed = True
                notes.append(
                    f"POL-ABU-010 violation: valor proposto R$ {refund:.2f} excede valor pago R$ {record.total_amount_brl:.2f}."
                )
                if "POL-ABU-010" not in citations:
                    citations.append("POL-ABU-010")

            # Rule 2: Abuse quarantine (POL-ABU-009)
            if (
                record.prior_refunds_30d >= abuse_limit
                and decision == ResolutionDecision.AUTO_APPROVED_REFUND
            ):
                revision_needed = True
                notes.append(
                    f"POL-ABU-009 violation: cliente possui {record.prior_refunds_30d} reembolsos em 30 dias (limite {abuse_limit}); autoaprovacao bloqueada."
                )
                if "POL-ABU-009" not in citations:
                    citations.append("POL-ABU-009")

        # Rule 3: Financial Governance Ceiling > R$ 150.00 requires Human-in-the-Loop (POL-FIN-004 / POL-FIN-150)
        if decision == ResolutionDecision.AUTO_APPROVED_REFUND and refund > threshold:
            revision_needed = True
            notes.append(
                f"POL-FIN-150 violation: reembolso de R$ {refund:.2f} excede alcada autonoma de R$ {threshold:.2f}; exigir ESCALATED_HITL."
            )
            if "POL-FIN-004" not in citations:
                citations.append("POL-FIN-004")

        # Guarantee convergence if max revisions already reached
        if revision_needed and state.revision_count >= 2:
            revision_needed = False
            if refund > threshold:
                decision = ResolutionDecision.ESCALATED_HITL
            elif record and record.prior_refunds_30d >= abuse_limit:
                decision = (
                    ResolutionDecision.ESCALATED_HITL
                    if record.missing_items_reported
                    else ResolutionDecision.REJECTED_POLICY
                )
                if decision == ResolutionDecision.REJECTED_POLICY:
                    refund = 0.0

        requires_hitl = decision == ResolutionDecision.ESCALATED_HITL or (
            refund > threshold and decision != ResolutionDecision.REJECTED_POLICY
        )

        if not revision_needed and not notes:
            notes.append(
                f"Validated against financial threshold (R$ {threshold:.2f}), order cap, and 30-day abuse policy."
            )

        verdict = JudgeVerdict(
            approved=not revision_needed,
            final_decision=decision,
            approved_refund_brl=refund if decision != ResolutionDecision.REJECTED_POLICY else 0.0,
            requires_human_in_the_loop=requires_hitl,
            revision_requested=revision_needed,
            reflection_notes=notes,
            policy_citations=citations,
        )
        state.judge_verdict = verdict
        if revision_needed:
            state.revision_count += 1

        prompt_tok = 220
        comp_tok = 75
        latency_ms = max(0.05, round((time.perf_counter() - t0) * 1000.0, 3))
        state.node_metrics.append(
            NodeTraceMetric(
                node_name="judge_validator_node",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                cost_usd=_estimate_cost(prompt_tok, comp_tok),
            )
        )

        if emit:
            emit(
                "judge_validated",
                {
                    "approved": verdict.approved,
                    "revision_requested": verdict.revision_requested,
                    "final_decision": verdict.final_decision.value,
                    "requires_human_in_the_loop": verdict.requires_human_in_the_loop,
                    "approved_refund_brl": verdict.approved_refund_brl,
                },
            )
        return state

    def _draft_initial_proposal(self, state: AgentGraphState) -> None:
        triage = state.triage
        record = state.operational_record
        intent = triage.intent if triage else IntentCategory.REFUND_REQUEST

        if record is None:
            retrieved_ids = [c.policy_id for c in state.retrieved_policies]
            state.draft_decision = ResolutionDecision.INFO_PROVIDED
            state.draft_refund_brl = 0.0
            state.draft_citations = retrieved_ids[:2]
            if triage and triage.extracted_order_id:
                state.draft_message = (
                    f"Nao localizamos o pedido {triage.extracted_order_id} na base operacional. "
                    "Confira o codigo do pedido (ex.: ORD-9001 a ORD-9015) para analise de SLA ou reembolso."
                )
            else:
                top_policy = state.retrieved_policies[0] if state.retrieved_policies else None
                policy_hint = (
                    f"Conforme a politica {top_policy.policy_id} ({top_policy.title}): {top_policy.content}"
                    if top_policy
                    else "Informe o codigo do pedido para verificacao operacional."
                )
                state.draft_message = policy_hint
            return

        # Operational synthesis grounded strictly in SQL record + retrieved policies
        abuse_limit = self.settings.max_refunds_30d_abuse_threshold

        # 1. Missing / Wrong Item disputes
        if record.missing_items_reported and intent in {
            IntentCategory.CATALOG_OR_ITEM_ISSUE,
            IntentCategory.REFUND_REQUEST,
        }:
            item_val = round(record.missing_items_value_brl, 2)
            state.draft_decision = ResolutionDecision.AUTO_APPROVED_REFUND
            state.draft_refund_brl = item_val
            state.draft_citations = ["POL-ITM-005", "POL-PAG-015"]
            state.draft_message = (
                f"Pedido {record.order_id} ({record.restaurant_name}): ausencia de item confirmada no valor de "
                f"R$ {item_val:.2f} (total do pedido R$ {record.total_amount_brl:.2f}). "
                f"Proposta inicial de estorno do item faltante de R$ {item_val:.2f} conforme POL-ITM-005."
            )
            return

        # 2. Pre-preparation cancellation
        if intent == IntentCategory.ORDER_CANCELLATION:
            if record.order_status in {"CREATED", "ACCEPTED"}:
                state.draft_decision = ResolutionDecision.AUTO_APPROVED_REFUND
                state.draft_refund_brl = round(record.total_amount_brl, 2)
                state.draft_citations = ["POL-CAN-007"]
                state.draft_message = (
                    f"Pedido {record.order_id} encontra-se em status {record.order_status} (antes do preparo). "
                    f"Elegivel para cancelamento integral de R$ {record.total_amount_brl:.2f} conforme POL-CAN-007."
                )
                return
            if record.delay_minutes < 45:
                state.draft_decision = ResolutionDecision.REJECTED_POLICY
                state.draft_refund_brl = 0.0
                state.draft_citations = ["POL-CAN-008"]
                state.draft_message = (
                    f"Cancelamento gratuito indisponivel para o pedido {record.order_id}: status atual "
                    f"{record.order_status} com atraso de {record.delay_minutes} min (inferior a 45 min), conforme POL-CAN-008."
                )
                return

        # 3. Still preparing / in transit within SLA
        if record.order_status in {"PREPARING", "IN_TRANSIT"} and record.delay_minutes < 15:
            state.draft_decision = ResolutionDecision.INFO_PROVIDED
            state.draft_refund_brl = 0.0
            state.draft_citations = ["POL-LOG-014"]
            state.draft_message = (
                f"Pedido {record.order_id} ({record.restaurant_name}) esta em status {record.order_status} "
                f"dentro do ETA prometido de {record.promised_eta_min} min (atraso atual {record.delay_minutes} min). "
                "Acompanhe o rastreamento em tempo real conforme POL-LOG-014."
            )
            return

        # 4. Severe Delay >= 45 min with GPS confirmation
        if record.delay_minutes >= 45 and record.courier_gps_delayed:
            # Respect user claimed cap if lower, else full order amount (and let Judge inspect >150 or >total)
            proposed = round(record.total_amount_brl, 2)
            if triage and triage.claimed_amount_brl and triage.claimed_amount_brl > record.total_amount_brl:
                proposed = round(triage.claimed_amount_brl, 2)
            state.draft_decision = ResolutionDecision.AUTO_APPROVED_REFUND
            state.draft_refund_brl = proposed
            state.draft_citations = ["POL-SLA-003", "POL-PAG-015"]
            state.draft_message = (
                f"Pedido {record.order_id} ({record.restaurant_name}): atraso severo confirmado de "
                f"{record.delay_minutes} min (ETA prometido {record.promised_eta_min} min, entregue em "
                f"{record.actual_delivery_min} min, GPS confirmado). Elegivel a reembolso integral de "
                f"R$ {record.total_amount_brl:.2f} conforme POL-SLA-003."
            )
            return

        # 5. Moderate Delay 30..44 min
        if 30 <= record.delay_minutes <= 44:
            if record.prior_refunds_30d >= abuse_limit:
                state.draft_decision = ResolutionDecision.REJECTED_POLICY
                state.draft_refund_brl = 0.0
                state.draft_citations = ["POL-ABU-009"]
                state.draft_message = (
                    f"Pedido {record.order_id}: cliente possui {record.prior_refunds_30d} reembolsos em 30 dias "
                    "(quarentena antifraude POL-ABU-009), impedindo concessao automatica."
                )
                return

            if record.customer_tier == "DIAMOND" and record.prior_refunds_30d <= 1:
                voucher = round(min(50.0, record.total_amount_brl * 0.35), 2)
                state.draft_decision = ResolutionDecision.PARTIAL_VOUCHER
                state.draft_refund_brl = voucher
                state.draft_citations = ["POL-TIER-011", "POL-SLA-002"]
                state.draft_message = (
                    f"Pedido {record.order_id} (Cliente DIAMOND): atraso moderado de {record.delay_minutes} min. "
                    f"Concedido voucher VIP de 35% no valor de R$ {voucher:.2f} conforme POL-TIER-011."
                )
            else:
                voucher = round(min(40.0, record.total_amount_brl * 0.25), 2)
                state.draft_decision = ResolutionDecision.PARTIAL_VOUCHER
                state.draft_refund_brl = voucher
                state.draft_citations = ["POL-SLA-002", "POL-TIER-012"]
                state.draft_message = (
                    f"Pedido {record.order_id} (Cliente {record.customer_tier}): atraso moderado de {record.delay_minutes} min. "
                    f"Concedido voucher compensatorio de 25% no valor de R$ {voucher:.2f} conforme POL-SLA-002."
                )
            return

        # 6. Mild Delay 15..29 min
        if 15 <= record.delay_minutes <= 29 and record.prior_refunds_30d < abuse_limit:
            state.draft_decision = ResolutionDecision.PARTIAL_VOUCHER
            state.draft_refund_brl = 15.0
            state.draft_citations = ["POL-SLA-001"]
            state.draft_message = (
                f"Pedido {record.order_id}: atraso leve de {record.delay_minutes} min (faixa 15-29 min). "
                "Nao ha reembolso integral, mas concedemos cupom compensatorio de R$ 15,00 valido por 14 dias (POL-SLA-001)."
            )
            return

        # 7. On-time delivery (< 15m delay, no missing item) or abuse flag
        citations = ["POL-LOG-013"]
        if record.prior_refunds_30d >= abuse_limit:
            citations.append("POL-ABU-009")
        state.draft_decision = ResolutionDecision.REJECTED_POLICY
        state.draft_refund_brl = 0.0
        state.draft_citations = citations
        state.draft_message = (
            f"Solicitacao de reembolso para o pedido {record.order_id} ({record.restaurant_name}) negada: "
            f"pedido entregue em {record.actual_delivery_min} min (ETA {record.promised_eta_min} min, "
            f"atraso de {record.delay_minutes} min < 15 min) sem ocorrencia de item faltante "
            f"e com {record.prior_refunds_30d} reembolsos nos ultimos 30 dias ({', '.join(citations)})."
        )

    def _apply_judge_revision(self, state: AgentGraphState) -> None:
        """Apply Judge Self-Reflection corrections to satisfy all financial & abuse policies."""
        record = state.operational_record
        threshold = self.settings.hitl_refund_threshold_brl
        abuse_limit = self.settings.max_refunds_30d_abuse_threshold
        citations = list(dict.fromkeys(state.draft_citations))

        if record and state.draft_refund_brl > record.total_amount_brl:
            state.draft_refund_brl = round(record.total_amount_brl, 2)
            if "POL-ABU-010" not in citations:
                citations.append("POL-ABU-010")

        if record and record.prior_refunds_30d >= abuse_limit:
            if "POL-ABU-009" not in citations:
                citations.append("POL-ABU-009")
            # Corrective RAG: ensure triggered antifraud policy chunk is grounded in context
            if all(c.policy_id != "POL-ABU-009" for c in state.retrieved_policies):
                extra, _ = self.retriever.retrieve("POL-ABU-009 abuso recorrente 4 reembolsos 30 dias", top_k_final=1)
                if extra:
                    state.retrieved_policies = (extra + state.retrieved_policies)[: self.settings.rag_top_k_reranked]
            if record.missing_items_reported:
                state.draft_decision = ResolutionDecision.ESCALATED_HITL
                state.draft_refund_brl = round(record.missing_items_value_brl, 2)
                state.draft_citations = citations
                state.draft_message = (
                    f"Pedido {record.order_id}: item faltante de R$ {record.missing_items_value_brl:.2f} reportado, "
                    f"porem a conta possui {record.prior_refunds_30d} reembolsos em 30 dias (POL-ABU-009). "
                    "Decisao revisada para ESCALATED_HITL para analise de risco humana."
                )
            else:
                state.draft_decision = ResolutionDecision.REJECTED_POLICY
                state.draft_refund_brl = 0.0
                state.draft_citations = citations
                state.draft_message = (
                    f"Pedido {record.order_id} bloqueado por politica antifraude (POL-ABU-009): "
                    f"{record.prior_refunds_30d} reembolsos nos ultimos 30 dias."
                )
            return

        if state.draft_refund_brl > threshold:
            if "POL-FIN-004" not in citations:
                citations.append("POL-FIN-004")
            # Corrective RAG: ensure POL-FIN-004 (POL-FIN-150) is grounded in retrieved context
            if all(c.policy_id != "POL-FIN-004" for c in state.retrieved_policies):
                extra, _ = self.retriever.retrieve("POL-FIN-150 alcada financeira 150 aprovacao humana HITL", top_k_final=1)
                if extra:
                    state.retrieved_policies = (extra + state.retrieved_policies)[: self.settings.rag_top_k_reranked]
            state.draft_decision = ResolutionDecision.ESCALATED_HITL
            state.draft_citations = citations
            order_ref = record.order_id if record else "solicitado"
            state.draft_message = (
                f"Pedido {order_ref}: elegivel ao estorno de R$ {state.draft_refund_brl:.2f}, "
                f"porem o valor ultrapassa a alcada autonoma de R$ {threshold:.2f} (protocolo POL-FIN-150 / POL-FIN-004). "
                "Encaminhado obrigatoriamente para aprovacao humana (Human-in-the-Loop - ESCALATED_HITL)."
            )
