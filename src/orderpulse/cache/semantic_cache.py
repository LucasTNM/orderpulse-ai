"""Semantic Cache layer using Redis Vector storage + cosine similarity (> 0.95, 24h TTL).

Intercepts semantically equivalent queries before invoking the multi-agent
LangGraph pipeline, dropping response latency to < 50ms and cutting LLM API
costs to zero on cache hits.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from orderpulse.config import Settings, get_settings
from orderpulse.rag.embeddings import DeterministicSemanticEmbedder, cosine_similarity
from orderpulse.schemas.models import AgentResolutionResponse

_ORDER_ID_RE = re.compile(r"\bORD-\d{4,8}\b", re.IGNORECASE)


@dataclass
class _CacheRecord:
    key: str
    order_scope: str
    query: str
    vector: np.ndarray
    response_json: str
    original_cost_usd: float
    expires_at: float


@dataclass
class CacheLookupResult:
    hit: bool
    similarity: Optional[float] = None
    response: Optional[AgentResolutionResponse] = None
    original_cost_usd: float = 0.0


class RedisSemanticCache:
    """Semantic cache with live Redis transport and zero-dependency in-memory fallback."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = DeterministicSemanticEmbedder(dims=384)
        self._memory_store: dict[str, list[_CacheRecord]] = {}
        self._redis_client = self._try_connect_redis()

    def _try_connect_redis(self) -> Any:
        try:
            import redis  # type: ignore[import-not-found]

            client = redis.Redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            )
            client.ping()
            return client
        except Exception:
            return None

    @property
    def backend_name(self) -> str:
        return "redis" if self._redis_client is not None else "in_memory_redis_fallback"

    @staticmethod
    def _extract_scope(query: str, order_id: Optional[str]) -> str:
        if order_id:
            return order_id.upper()
        match = _ORDER_ID_RE.search(query)
        return match.group(0).upper() if match else "__GLOBAL__"

    def _embed_intent_query(self, query: str) -> np.ndarray:
        """Embed query text stripped of order IDs so cosine measures pure intent similarity."""
        cleaned = _ORDER_ID_RE.sub(" ", query)
        return self.embedder.embed(cleaned)

    def lookup(
        self,
        query: str,
        order_id: Optional[str] = None,
    ) -> CacheLookupResult:
        """Return a cached response when cosine similarity exceeds threshold within TTL."""
        if not self.settings.semantic_cache_enabled:
            return CacheLookupResult(hit=False)

        scope = self._extract_scope(query, order_id)
        query_vec = self._embed_intent_query(query)
        threshold = self.settings.semantic_cache_threshold
        now = time.time()

        best_sim = 0.0
        best_record: Optional[_CacheRecord] = None

        if self._redis_client is not None:
            candidates = self._load_from_redis(scope)
        else:
            candidates = self._memory_store.get(scope, [])
            # Evict expired records lazily
            candidates = [r for r in candidates if r.expires_at > now]
            self._memory_store[scope] = candidates

        for rec in candidates:
            sim = cosine_similarity(query_vec, rec.vector)
            if sim > best_sim:
                best_sim = sim
                best_record = rec

        if best_record is not None and best_sim > threshold:
            parsed = AgentResolutionResponse.model_validate_json(best_record.response_json)
            return CacheLookupResult(
                hit=True,
                similarity=round(best_sim, 4),
                response=parsed,
                original_cost_usd=best_record.original_cost_usd,
            )

        return CacheLookupResult(
            hit=False,
            similarity=round(best_sim, 4) if best_sim > 0.0 else None,
        )

    def store(
        self,
        query: str,
        response: AgentResolutionResponse,
        order_id: Optional[str] = None,
    ) -> None:
        """Persist a verified resolution response with 24-hour TTL."""
        if not self.settings.semantic_cache_enabled:
            return
        # Never cache security guardrail blocks so attacks are always logged/audited fresh
        if response.decision.value == "BLOCKED_GUARDRAIL":
            return

        scope = self._extract_scope(query, order_id or response.triage.extracted_order_id)
        vec = self._embed_intent_query(query)
        ttl = self.settings.semantic_cache_ttl_seconds
        key = f"orderpulse:semcache:{scope}:{uuid.uuid4().hex[:12]}"
        payload_json = response.model_dump_json()

        if self._redis_client is not None:
            try:
                redis_doc = json.dumps(
                    {
                        "query": query,
                        "vector": vec.tolist(),
                        "response_json": payload_json,
                        "original_cost_usd": response.total_cost_usd,
                    }
                )
                self._redis_client.setex(key, ttl, redis_doc)
                return
            except Exception:
                pass

        record = _CacheRecord(
            key=key,
            order_scope=scope,
            query=query,
            vector=vec,
            response_json=payload_json,
            original_cost_usd=response.total_cost_usd,
            expires_at=time.time() + ttl,
        )
        self._memory_store.setdefault(scope, []).append(record)

    def _load_from_redis(self, scope: str) -> list[_CacheRecord]:
        records: list[_CacheRecord] = []
        try:
            pattern = f"orderpulse:semcache:{scope}:*"
            for key in self._redis_client.scan_iter(match=pattern, count=100):
                raw = self._redis_client.get(key)
                if not raw:
                    continue
                data = json.loads(raw)
                records.append(
                    _CacheRecord(
                        key=key,
                        order_scope=scope,
                        query=data["query"],
                        vector=np.array(data["vector"], dtype=np.float32),
                        response_json=data["response_json"],
                        original_cost_usd=float(data.get("original_cost_usd", 0.0)),
                        expires_at=time.time() + 1.0,
                    )
                )
        except Exception:
            return self._memory_store.get(scope, [])
        return records

    def clear(self) -> None:
        """Flush in-memory semantic cache entries (useful for isolated evaluations)."""
        self._memory_store.clear()
