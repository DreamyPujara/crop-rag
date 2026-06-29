"""
Retrieval Pipeline
- Semantic-only (via VectorStore)
- BM25 keyword retrieval
- Hybrid = BM25 + Semantic fused with Reciprocal Rank Fusion (RRF)
- Query rewriting for agronomic shorthand
- Optional CrossEncoder reranker
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.vector_store import VectorStore


# ── Query rewriter ────────────────────────────────────────────────────────────

AGRONOMIC_EXPANSIONS = {
    r"\bdry weather\b":        "low rainfall high temperature",
    r"\bwet climate\b":        "high rainfall high humidity",
    r"\btropical\b":           "high temperature high humidity",
    r"\bacidic soil\b":        "low pH acidic",
    r"\balkaline soil\b":      "high pH alkaline",
    r"\bfertile soil\b":       "high nitrogen high phosphorus high potassium",
    r"\bwaterlogged\b":        "high rainfall waterlogged flooding",
    r"\bdrought.?resistant\b": "low rainfall drought",
    r"\bhumid\b":              "high humidity",
    r"\bhigh N\b":             "high nitrogen",
    r"\blow N\b":              "low nitrogen",
}


def rewrite_query(query: str) -> str:
    """Expand agronomic shorthand before retrieval."""
    q = query.lower()
    for pattern, expansion in AGRONOMIC_EXPANSIONS.items():
        q = re.sub(pattern, expansion, q, flags=re.IGNORECASE)
    return q


# ── BM25 index ────────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        tokenized = [d["text"].lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {**self.docs[i], "score": float(scores[i])}
            for i in top_idx
        ]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Merge multiple ranked lists via RRF. k=60 is the standard constant."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = doc.get("metadata", {}).get("chunk_type", "") + doc["text"][:40]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

    merged = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return [{**doc_map[d], "score": scores[d]} for d in merged]


# ── Main Retriever ────────────────────────────────────────────────────────────

class Retriever:
    """
    Unified retriever supporting semantic, BM25, and hybrid modes.
    Optionally applies query rewriting and CrossEncoder reranking.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        docs: List[Dict[str, Any]],
        mode: str = "hybrid",          # "semantic" | "bm25" | "hybrid"
        use_rewriter: bool = True,
        use_reranker: bool = False,
    ):
        self.vs = vector_store
        self.mode = mode
        self.use_rewriter = use_rewriter
        self.use_reranker = use_reranker
        self.bm25_index = BM25Index(docs)
        self.reranker: Optional[CrossEncoder] = None

        if use_reranker:
            print("Loading CrossEncoder reranker …")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_rewriter:
            query = rewrite_query(query)

        fetch_k = max(k * 2, 10)  # fetch more for fusion/reranking

        if self.mode == "semantic":
            results = self.vs.search(query, k=fetch_k, metadata_filter=metadata_filter)

        elif self.mode == "bm25":
            results = self.bm25_index.search(query, k=fetch_k)

        else:  # hybrid
            sem_results = self.vs.search(query, k=fetch_k, metadata_filter=metadata_filter)
            bm25_results = self.bm25_index.search(query, k=fetch_k)
            results = reciprocal_rank_fusion([sem_results, bm25_results])

        # Rerank top-10 → top-k
        if self.use_reranker and self.reranker and len(results) > 0:
            pairs = [(query, r["text"]) for r in results[:10]]
            ce_scores = self.reranker.predict(pairs)
            ranked = sorted(
                zip(results[:10], ce_scores), key=lambda x: x[1], reverse=True
            )
            results = [r for r, _ in ranked]

        return results[:k]
