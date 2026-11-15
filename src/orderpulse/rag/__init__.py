"""Hybrid RAG (BM25 + Dense Vector + RRF + Cross-Encoder Reranker) package."""

from orderpulse.rag.embeddings import DeterministicSemanticEmbedder, cosine_similarity
from orderpulse.rag.hybrid_retriever import HybridPolicyRetriever
from orderpulse.rag.policies_corpus import POLICY_CORPUS

__all__ = [
    "DeterministicSemanticEmbedder",
    "cosine_similarity",
    "HybridPolicyRetriever",
    "POLICY_CORPUS",
]
