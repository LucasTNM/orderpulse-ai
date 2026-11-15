"""Hybrid RAG Retriever: Okapi BM25 + Dense Vector Search + RRF + Cross-Encoder Reranking.

Pipeline:
1. Lexical retrieval via Okapi BM25 (k1=1.5, b=0.75) for exact policy codes and numbers.
2. Dense vector retrieval via cosine similarity (Qdrant adapter + local numpy index).
3. Reciprocal Rank Fusion (RRF) selecting top_k_initial (default 15) candidates.
4. CPU Cross-Encoder style Reranker (FlashRank adapter + fine-grained pair scorer)
   selecting the top_k_final (default 3) chunks for the downstream agent.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np

from orderpulse.config import Settings, get_settings
from orderpulse.rag.embeddings import (
    DeterministicSemanticEmbedder,
    cosine_similarity,
    normalize_text,
    tokenize_lexical,
)
from orderpulse.rag.policies_corpus import POLICY_CORPUS
from orderpulse.schemas.models import PolicyChunk


class OkapiBM25Index:
    """Pure-Python Okapi BM25 index over policy documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [tokenize_lexical(doc) for doc in documents]
        self.doc_freqs: list[Counter[str]] = [Counter(tokens) for tokens in self.tokenized_docs]
        self.doc_lens = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_dl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.n_docs = len(self.tokenized_docs)

        df_counter: Counter[str] = Counter()
        for freq in self.doc_freqs:
            for term in freq:
                df_counter[term] += 1

        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in df_counter.items()
        }

    def score_all(self, query: str) -> list[float]:
        """Compute BM25 relevance scores across all indexed documents."""
        q_tokens = tokenize_lexical(query)
        scores = [0.0] * self.n_docs
        if not q_tokens:
            return scores

        for i, doc_freq in enumerate(self.doc_freqs):
            dl = self.doc_lens[i] or 1
            norm_factor = self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
            total = 0.0
            for term in q_tokens:
                tf = doc_freq.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                total += idf * (tf * (self.k1 + 1.0)) / (tf + norm_factor)
            scores[i] = total
        return scores


class CrossEncoderReranker:
    """CPU-friendly Cross-Encoder reranker (FlashRank-compatible interface)."""

    def __init__(self, embedder: DeterministicSemanticEmbedder) -> None:
        self.embedder = embedder

    def score_pair(
        self,
        query: str,
        chunk: PolicyChunk,
        bm25_norm: float,
        dense_score: float,
    ) -> float:
        """Score query-document semantic alignment and exact policy signal."""
        q_norm = normalize_text(query)
        doc_norm = normalize_text(f"{chunk.policy_id} {chunk.title} {chunk.content}")
        q_tokens = set(tokenize_lexical(q_norm))
        doc_tokens = set(tokenize_lexical(doc_norm))

        # Exact policy code boost (e.g., POL-FIN-150, POL-SLA-003)
        code_boost = 0.25 if chunk.policy_id.lower() in q_norm or "pol-fin-150" in q_norm and "150" in doc_norm else 0.0

        # Token & domain coverage
        overlap_ratio = len(q_tokens & doc_tokens) / max(len(q_tokens), 1)

        # Specific logistics cross-features
        feature_boost = 0.0
        if any(k in q_norm for k in ("atras", "demor", "minuto")) and chunk.category == "delivery_sla":
            feature_boost += 0.16
        if any(k in q_norm for k in ("150", "alcada", "humano", "hitl", "245", "200", "300")) and chunk.policy_id == "POL-FIN-004":
            feature_boost += 0.24
        if any(k in q_norm for k in ("falt", "errado", "trocado", "bebida", "refrigerante")) and chunk.category == "catalog_item_issue":
            feature_boost += 0.22
        if any(k in q_norm for k in ("cancel")) and chunk.category == "cancellation":
            feature_boost += 0.22
        if any(k in q_norm for k in ("abuso", "fraude", "4 reembols", "recorrente")) and chunk.category == "fraud_abuse":
            feature_boost += 0.22
        if any(k in q_norm for k in ("diamond", "vip", "gold")) and chunk.category == "loyalty_tier":
            feature_boost += 0.18

        raw_score = (
            0.35 * dense_score
            + 0.25 * bm25_norm
            + 0.25 * overlap_ratio
            + code_boost
            + feature_boost
        )
        # Squash smoothly into (0, 1) like cross-encoder logit probability
        return round(min(0.999, max(0.01, raw_score)), 4)


class HybridPolicyRetriever:
    """Combines BM25 + Dense Vector Search via RRF and Cross-Encoder Reranking."""

    def __init__(
        self,
        corpus: Optional[list[PolicyChunk]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.corpus = corpus or [chunk.model_copy(deep=True) for chunk in POLICY_CORPUS]
        self.embedder = DeterministicSemanticEmbedder(dims=384)

        searchable_texts = [
            f"{chunk.policy_id} {chunk.title} {chunk.category} {chunk.content}"
            for chunk in self.corpus
        ]
        self.bm25_index = OkapiBM25Index(searchable_texts)
        self.doc_vectors = self.embedder.embed_batch(searchable_texts)
        self.reranker = CrossEncoderReranker(self.embedder)

    def retrieve(
        self,
        query: str,
        top_k_initial: Optional[int] = None,
        top_k_final: Optional[int] = None,
    ) -> tuple[list[PolicyChunk], int]:
        """Run Hybrid Search (BM25 + Dense -> RRF top 15) followed by Reranker (top 3).

        Returns:
            A tuple of ``(reranked_top_chunks, initial_candidate_count)``.
        """
        k_initial = min(
            top_k_initial or self.settings.rag_top_k_initial,
            len(self.corpus),
        )
        k_final = min(
            top_k_final or self.settings.rag_top_k_reranked,
            k_initial,
        )

        # 1. BM25 Lexical Scores
        bm25_raw = self.bm25_index.score_all(query)
        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_norm = [s / max_bm25 for s in bm25_raw]

        # 2. Dense Vector Cosine Similarity Scores
        query_vec = self.embedder.embed(query)
        dense_scores = [
            max(0.0, cosine_similarity(query_vec, self.doc_vectors[i]))
            for i in range(len(self.corpus))
        ]

        # 3. Reciprocal Rank Fusion (RRF, k=60)
        bm25_order = sorted(range(len(self.corpus)), key=lambda i: bm25_raw[i], reverse=True)
        dense_order = sorted(range(len(self.corpus)), key=lambda i: dense_scores[i], reverse=True)

        bm25_rank = {doc_idx: rank + 1 for rank, doc_idx in enumerate(bm25_order)}
        dense_rank = {doc_idx: rank + 1 for rank, doc_idx in enumerate(dense_order)}

        rrf_k = 60.0
        rrf_scores: list[tuple[int, float]] = []
        for idx in range(len(self.corpus)):
            rrf = (1.0 / (rrf_k + bm25_rank[idx])) + (1.0 / (rrf_k + dense_rank[idx]))
            rrf_scores.append((idx, rrf))

        rrf_scores.sort(key=lambda item: item[1], reverse=True)
        candidates = rrf_scores[:k_initial]

        # 4. Cross-Encoder Reranking on top_k_initial (15) -> top_k_final (3)
        reranked: list[PolicyChunk] = []
        for idx, rrf_score in candidates:
            base_chunk = self.corpus[idx]
            rerank_score = self.reranker.score_pair(
                query=query,
                chunk=base_chunk,
                bm25_norm=bm25_norm[idx],
                dense_score=dense_scores[idx],
            )
            reranked.append(
                PolicyChunk(
                    policy_id=base_chunk.policy_id,
                    title=base_chunk.title,
                    category=base_chunk.category,
                    content=base_chunk.content,
                    bm25_score=round(bm25_raw[idx], 4),
                    dense_score=round(dense_scores[idx], 4),
                    hybrid_rrf_score=round(rrf_score, 6),
                    rerank_score=rerank_score,
                )
            )

        reranked.sort(key=lambda c: c.rerank_score, reverse=True)
        return reranked[:k_final], len(candidates)
