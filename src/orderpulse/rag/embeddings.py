"""Deterministic domain-calibrated semantic embedder for Hybrid RAG and Semantic Caching.

Produces L2-normalized dense vectors (384 dimensions) combining canonicalized
domain concepts, Portuguese/English stemming, bigrams, and character n-grams.
Optionally delegates to OpenAI embeddings when remote LLM mode is enabled.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_-]+")
_ORDER_RE = re.compile(r"\bord-\d{4,8}\b")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "o",
        "as",
        "os",
        "um",
        "uma",
        "uns",
        "umas",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "por",
        "para",
        "pra",
        "com",
        "sem",
        "que",
        "e",
        "ou",
        "se",
        "meu",
        "minha",
        "meus",
        "minhas",
        "seu",
        "sua",
        "ola",
        "oi",
        "porfavor",
        "favor",
        "ja",
        "ha",
        "faz",
        "esta",
        "estao",
        "foi",
        "sera",
        "muito",
        "mais",
        "sobre",
        "chegou",
        "alem",
        "ficou",
        "teve",
        "tinha",
        "estava",
        "esse",
        "este",
        "essa",
        "agora",
        "hoje",
        "aqui",
        "the",
        "is",
        "are",
        "to",
        "of",
        "and",
        "in",
        "for",
        "my",
        "this",
        "that",
    }
)

# Canonicalize logistics/delivery synonyms so paraphrases align closely in vector space.
_CONCEPT_MAP: dict[str, str] = {
    "atrasado": "concept_delay",
    "atrasada": "concept_delay",
    "atraso": "concept_delay",
    "atrasou": "concept_delay",
    "demora": "concept_delay",
    "demorando": "concept_delay",
    "demorou": "concept_delay",
    "tarde": "concept_delay",
    "passou": "concept_delay",
    "prazo": "concept_delay",
    "sla": "concept_sla",
    "eta": "concept_sla",
    "reembolso": "concept_refund",
    "reembolsar": "concept_refund",
    "estorno": "concept_refund",
    "estornar": "concept_refund",
    "ressarcimento": "concept_refund",
    "devolucao": "concept_refund",
    "devolver": "concept_refund",
    "cupom": "concept_voucher",
    "voucher": "concept_voucher",
    "credito": "concept_voucher",
    "cancelar": "concept_cancel",
    "cancelamento": "concept_cancel",
    "cancele": "concept_cancel",
    "desistir": "concept_cancel",
    "faltou": "concept_missing_item",
    "faltando": "concept_missing_item",
    "faltante": "concept_missing_item",
    "incompleto": "concept_missing_item",
    "errado": "concept_wrong_item",
    "trocado": "concept_wrong_item",
    "aprovacao": "concept_hitl",
    "humana": "concept_hitl",
    "humano": "concept_hitl",
    "alcada": "concept_hitl",
    "hitl": "concept_hitl",
    "fraude": "concept_abuse",
    "abuso": "concept_abuse",
    "recorrente": "concept_abuse",
    "historico": "concept_history",
    "cliente": "concept_customer",
    "pedido": "concept_order",
    "entrega": "concept_delivery",
    "entregador": "concept_courier",
    "motoboy": "concept_courier",
    "minutos": "concept_minutes",
    "minuto": "concept_minutes",
    "min": "concept_minutes",
    "quero": "concept_request",
    "preciso": "concept_request",
    "solicito": "concept_request",
    "gostaria": "concept_request",
    "peco": "concept_request",
    "valor": "concept_amount",
    "reais": "concept_currency",
    "brl": "concept_currency",
}


def normalize_text(text: str) -> str:
    """Lowercase and strip accents while keeping alphanumeric, hyphens, and underscores."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize_lexical(text: str) -> list[str]:
    """Tokenize text for BM25 preserving exact codes (e.g., pol-fin-150) and numbers."""
    norm = normalize_text(text)
    raw_tokens = _TOKEN_RE.findall(norm)
    tokens: list[str] = []
    for tok in raw_tokens:
        if len(tok) <= 1 and not tok.isdigit():
            continue
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)
        if tok in _CONCEPT_MAP:
            tokens.append(_CONCEPT_MAP[tok])
    return tokens


def _hash_feature(feature: str, dims: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, "little")
    idx = val % dims
    sign = 1.0 if ((val >> 63) & 1) == 0 else -1.0
    return idx, sign


class DeterministicSemanticEmbedder:
    """384-d L2-normalized semantic embedder for Hybrid RAG & Redis Semantic Cache."""

    def __init__(self, dims: int = 384) -> None:
        self.dims = dims

    def embed(self, text: str) -> np.ndarray:
        """Encode text into a unit-norm float32 numpy vector."""
        vec = np.zeros(self.dims, dtype=np.float32)
        norm = normalize_text(text)
        raw_tokens = _TOKEN_RE.findall(norm)

        # Anchor explicit order IDs heavily so queries for distinct orders do not collide.
        for order_id in _ORDER_RE.findall(norm):
            for salt in ("ord_a", "ord_b", "ord_c"):
                idx, sign = _hash_feature(f"{salt}:{order_id}", self.dims)
                vec[idx] += sign * 4.0

        meaningful: list[str] = []
        seen_concepts: set[str] = set()
        for tok in raw_tokens:
            if tok in _STOPWORDS:
                continue
            concept = _CONCEPT_MAP.get(tok)
            if concept is not None:
                meaningful.append(concept)
                if concept not in seen_concepts:
                    seen_concepts.add(concept)
                    idx, sign = _hash_feature(f"cpt:{concept}", self.dims)
                    vec[idx] += sign * 4.5
            else:
                stem = tok[:5] if len(tok) > 5 and not tok.isdigit() else tok
                meaningful.append(stem)
                weight = 2.0 if tok.isdigit() else 1.1
                idx, sign = _hash_feature(f"tok:{stem}", self.dims)
                vec[idx] += sign * weight

                # Subword 4-grams capture morphological similarity across Portuguese variations.
                if len(tok) >= 4 and not tok.isdigit():
                    for i in range(len(tok) - 3):
                        sub = tok[i : i + 4]
                        s_idx, s_sign = _hash_feature(f"chr:{sub}", self.dims)
                        vec[s_idx] += s_sign * 0.20

        # Bigrams of canonicalized concepts capture phrasing intent.
        for a, b in zip(meaningful, meaningful[1:]):
            idx, sign = _hash_feature(f"bi:{a}|{b}", self.dims)
            vec[idx] += sign * 0.25

        l2 = float(np.linalg.norm(vec))
        if l2 > 0.0:
            vec /= l2
        return vec

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        """Encode an iterable of strings into an (N, dims) float32 matrix."""
        rows = [self.embed(t) for t in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.dims), dtype=np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute bounded cosine similarity in [-1.0, 1.0] between two vectors."""
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0 or math.isnan(denom):
        return 0.0
    score = float(np.dot(vec_a, vec_b) / denom)
    return max(-1.0, min(1.0, score))
