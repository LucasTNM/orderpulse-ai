"""Intent Router & Input Guardrail using strict Pydantic structured output.

Protects downstream LangGraph nodes from Prompt Injection, SQL Injection,
financial override attacks, and PII leakage while classifying user intent.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from orderpulse.config import Settings, get_settings
from orderpulse.schemas.models import (
    GuardrailStatus,
    IntentCategory,
    OrderResolveRequest,
    TriageClassification,
)

_ORDER_ID_PATTERN = re.compile(r"\b(ORD-\d{4,8})\b", re.IGNORECASE)
_CURRENCY_PATTERN = re.compile(r"(?:r\$|brl)\s*(\d+(?:[.,]\d{1,2})?)", re.IGNORECASE)

_PROMPT_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Instruction override attempt",
        re.compile(
            r"(ignore|desconsidere|esque[cç]a|override|bypass|burl[ae]r?)\s+"
            r"(all|todas|as|previous|anteriores|suas|regras|rules|instructions|instru[cç][oõ]es|pol[ií]ticas)",
            re.IGNORECASE,
        ),
    ),
    (
        "System prompt or jailbreak extraction",
        re.compile(
            r"(system\s*prompt|developer\s*mode|jailbreak|dan\s*mode|modo\s*desenvolvedor|"
            r"revele\s*(o|seu)\s*prompt|print\s*instructions)",
            re.IGNORECASE,
        ),
    ),
    (
        "SQL injection / DDL manipulation attempt",
        re.compile(
            r"(;\s*(drop|delete|truncate|update|insert|alter)\s+|union\s+select|"
            r"\bdrop\s+table\b|\bdelete\s+from\b|--\s*$)",
            re.IGNORECASE,
        ),
    ),
    (
        "Direct financial guardrail bypass directive",
        re.compile(
            r"(aprove\s*(imediatamente|sem\s*validar|automaticamente)\s*(um\s*)?reembolso|"
            r"libere\s*(o\s*)?estorno\s*sem\s*hitl|"
            r"force\s*auto[_\s-]*approve)",
            re.IGNORECASE,
        ),
    ),
)

_PII_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(me\s*(passe|d[eê]|informe|envie)|liste|mostre)\s*(o\s*)?"
        r"(cpf|cart[aã]o\s*de\s*cr[eé]dito|cvv|senha|endere[cç]o\s*do\s*entregador|dados\s*banc[aá]rios)\s*"
        r"(de\s*outro|do\s*cliente|de\s*todos)",
        re.IGNORECASE,
    ),
)


class IntentRouterGuardrail:
    """Routes incoming customer queries and enforces security guardrails.

    Supports external LLM structured-output calls (Groq / OpenAI via httpx)
    when ``llm_provider`` is configured with valid API keys, while running a
    high-precision deterministic classifier by default or on fallback.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def classify(self, request: OrderResolveRequest) -> TriageClassification:
        """Run guardrail checks first, then classify intent via structured schema."""
        guardrail_status, guardrail_reason = self._check_guardrails(request.query)
        extracted_order_id = self._extract_order_id(request)
        claimed_amount = self._extract_claimed_amount(request.query)

        if guardrail_status != GuardrailStatus.SAFE:
            return TriageClassification(
                intent=IntentCategory.FRAUD_OR_SECURITY_ALERT,
                guardrail_status=guardrail_status,
                guardrail_reason=guardrail_reason,
                extracted_order_id=extracted_order_id,
                claimed_amount_brl=claimed_amount,
                urgency_score=5,
                requires_sql_lookup=False,
                requires_policy_rag=False,
                confidence=0.99,
            )

        if self._can_use_remote_llm():
            remote_result = self._classify_with_remote_llm(
                request=request,
                extracted_order_id=extracted_order_id,
                claimed_amount=claimed_amount,
            )
            if remote_result is not None:
                return remote_result

        return self._classify_deterministic(
            query=request.query,
            extracted_order_id=extracted_order_id,
            claimed_amount=claimed_amount,
        )

    def _check_guardrails(self, query: str) -> tuple[GuardrailStatus, str]:
        for label, pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(query):
                return (
                    GuardrailStatus.PROMPT_INJECTION_DETECTED,
                    f"Blocked by Input Guardrail: {label}.",
                )

        for pattern in _PII_LEAK_PATTERNS:
            if pattern.search(query):
                return (
                    GuardrailStatus.PII_LEAK_RISK,
                    "Blocked by Input Guardrail: unauthorized third-party PII request.",
                )

        return GuardrailStatus.SAFE, "Passed input security and prompt injection checks."

    @staticmethod
    def _extract_order_id(request: OrderResolveRequest) -> Optional[str]:
        if request.order_id:
            return request.order_id.upper()
        match = _ORDER_ID_PATTERN.search(request.query)
        return match.group(1).upper() if match else None

    @staticmethod
    def _extract_claimed_amount(query: str) -> Optional[float]:
        match = _CURRENCY_PATTERN.search(query)
        if not match:
            return None
        normalized = match.group(1).replace(",", ".")
        try:
            return round(float(normalized), 2)
        except ValueError:
            return None

    def _classify_deterministic(
        self,
        query: str,
        extracted_order_id: Optional[str],
        claimed_amount: Optional[float],
    ) -> TriageClassification:
        q = query.lower()

        cancel_kw = ("cancelar", "cancelamento", "desistir", "cancele", "abortar pedido")
        item_kw = (
            "faltando",
            "faltou",
            "item errado",
            "veio errado",
            "trocado",
            "incompleto",
            "sem bebida",
            "sem refrigerante",
            "derramado",
            "violado",
            "cardápio",
            "catálogo",
            "alérgeno",
        )
        delay_kw = (
            "atrasado",
            "atraso",
            "demorando",
            "demora",
            "não chegou",
            "nao chegou",
            "passou do prazo",
            "eta",
            "entrega parada",
            "entregador",
        )
        refund_kw = (
            "reembolso",
            "estorno",
            "devolver dinheiro",
            "dinheiro de volta",
            "cupom",
            "ressarcimento",
            "cobrança",
            "reembolsar",
        )
        out_of_scope_kw = (
            "previsão do tempo",
            "receita de bolo",
            "futebol",
            "código python",
            "piada",
            "bitcoin",
        )

        if any(kw in q for kw in out_of_scope_kw) and extracted_order_id is None:
            return TriageClassification(
                intent=IntentCategory.OUT_OF_SCOPE,
                guardrail_status=GuardrailStatus.SAFE,
                guardrail_reason="Passed security check; classified as out-of-domain query.",
                extracted_order_id=None,
                claimed_amount_brl=None,
                urgency_score=1,
                requires_sql_lookup=False,
                requires_policy_rag=False,
                confidence=0.93,
            )

        if any(kw in q for kw in cancel_kw):
            intent = IntentCategory.ORDER_CANCELLATION
            urgency = 4
            confidence = 0.94
        elif any(kw in q for kw in item_kw):
            intent = IntentCategory.CATALOG_OR_ITEM_ISSUE
            urgency = 4
            confidence = 0.93
        elif any(kw in q for kw in delay_kw):
            intent = IntentCategory.DELIVERY_DELAY_DISPUTE
            urgency = 4 if ("1 hora" in q or "urgente" in q or "frio" in q) else 3
            confidence = 0.95
        elif any(kw in q for kw in refund_kw):
            intent = IntentCategory.REFUND_REQUEST
            urgency = 4 if (claimed_amount or 0.0) > self.settings.hitl_refund_threshold_brl else 3
            confidence = 0.94
        elif extracted_order_id is not None:
            intent = IntentCategory.DELIVERY_DELAY_DISPUTE
            urgency = 3
            confidence = 0.85
        else:
            intent = IntentCategory.REFUND_REQUEST
            urgency = 2
            confidence = 0.80

        return TriageClassification(
            intent=intent,
            guardrail_status=GuardrailStatus.SAFE,
            guardrail_reason="Passed input security and prompt injection checks.",
            extracted_order_id=extracted_order_id,
            claimed_amount_brl=claimed_amount,
            urgency_score=urgency,
            requires_sql_lookup=extracted_order_id is not None,
            requires_policy_rag=True,
            confidence=confidence,
        )

    def _can_use_remote_llm(self) -> bool:
        provider = self.settings.llm_provider.lower()
        if provider == "openai" and self.settings.openai_api_key:
            return True
        if provider == "groq" and self.settings.groq_api_key:
            return True
        return False

    def _classify_with_remote_llm(
        self,
        request: OrderResolveRequest,
        extracted_order_id: Optional[str],
        claimed_amount: Optional[float],
    ) -> Optional[TriageClassification]:
        """Call OpenAI or Groq chat completions with strict JSON schema validation."""
        provider = self.settings.llm_provider.lower()
        if provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            api_key = self.settings.groq_api_key
        else:
            url = "https://api.openai.com/v1/chat/completions"
            api_key = self.settings.openai_api_key

        system_prompt = (
            "You are the OrderPulse AI Triage Router. Return strictly valid JSON matching "
            "TriageClassification with intent in "
            "[DELIVERY_DELAY_DISPUTE, REFUND_REQUEST, ORDER_CANCELLATION, "
            "CATALOG_OR_ITEM_ISSUE, FRAUD_OR_SECURITY_ALERT, OUT_OF_SCOPE], "
            "guardrail_status='SAFE', urgency_score (1..5), requires_sql_lookup (bool), "
            "requires_policy_rag (bool), confidence (0..1)."
        )
        payload = {
            "model": self.settings.llm_model_router,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": request.query,
                            "order_id": extracted_order_id,
                            "claimed_amount_brl": claimed_amount,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        try:
            with httpx.Client(timeout=8.0) as client:
                response = client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                raw_content = response.json()["choices"][0]["message"]["content"]
                parsed = json.loads(raw_content)
                parsed.setdefault("extracted_order_id", extracted_order_id)
                parsed.setdefault("claimed_amount_brl", claimed_amount)
                parsed.setdefault("guardrail_status", GuardrailStatus.SAFE.value)
                parsed.setdefault("guardrail_reason", "Validated via LLM router.")
                return TriageClassification.model_validate(parsed)
        except Exception:
            return None
