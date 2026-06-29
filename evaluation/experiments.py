"""
Experiment Runner
Systematically compares:
  - Chunking strategies (row vs aggregate vs hybrid)
  - Embedding models (minilm vs mpnet)
  - Retrieval modes (semantic vs bm25 vs hybrid)
  - k values (3, 5, 10)
  - With/without reranker

Results are saved to evaluation/experiment_results.json
"""

import json
import os
import time
from typing import List, Dict, Any
import numpy as np

from src.ingestion import load_and_chunk
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import generate_answer
from evaluation.evaluator import (
    TEST_QA_PAIRS, compute_retrieval_metrics,
    extract_crops_from_hits, llm_judge,
)

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DATA_PATH = "data/Crop_recommendation.csv"


def quick_eval(retriever, k: int = 5, n: int = 10) -> Dict[str, float]:
    """Fast retrieval-only eval on first n questions."""
    metrics_list = []
    for qa in TEST_QA_PAIRS[:n]:
        hits = retriever.retrieve(qa["question"], k=k)
        crops = extract_crops_from_hits(hits)
        m = compute_retrieval_metrics(crops, qa["expected_crops"], k=k)
        metrics_list.append(m)

    return {
        "precision_k": np.mean([m["precision_k"] for m in metrics_list]),
        "recall_k":    np.mean([m["recall_k"]    for m in metrics_list]),
        "hit_rate":    np.mean([m["hit_rate"]     for m in metrics_list]),
        "mrr":         np.mean([m["rr"]           for m in metrics_list]),
    }


def run_all_experiments():
    results = []

    # ── Exp 1: Chunking strategy comparison ──────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 1: Chunking Strategies")
    print("="*60)
    for strategy in ["row", "aggregate", "hybrid"]:
        docs = load_and_chunk(DATA_PATH, strategy=strategy)
        vs = VectorStore(
            collection_name=f"exp_{strategy}",
            model_key="minilm",
            persist_dir=f"./chroma_exp/{strategy}",
        )
        vs.index(docs)
        ret = Retriever(vs, docs, mode="semantic", use_rewriter=True, use_reranker=False)
        m = quick_eval(ret, k=5)
        row = {"experiment": "chunking", "variant": strategy, "k": 5, **m}
        results.append(row)
        print(f"  {strategy:12s} → hit_rate={m['hit_rate']:.3f}  recall={m['recall_k']:.3f}  MRR={m['mrr']:.3f}")

    # ── Exp 2: Embedding model comparison ────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 2: Embedding Models")
    print("="*60)
    docs = load_and_chunk(DATA_PATH, strategy="hybrid")
    for model_key in ["minilm", "mpnet"]:
        vs = VectorStore(
            collection_name=f"exp_emb_{model_key}",
            model_key=model_key,
            persist_dir=f"./chroma_exp/emb_{model_key}",
        )
        vs.index(docs)
        ret = Retriever(vs, docs, mode="semantic", use_rewriter=True, use_reranker=False)
        m = quick_eval(ret, k=5)
        row = {"experiment": "embedding", "variant": model_key, "k": 5, **m}
        results.append(row)
        print(f"  {model_key:10s} → hit_rate={m['hit_rate']:.3f}  recall={m['recall_k']:.3f}  MRR={m['mrr']:.3f}")

    # ── Exp 3: Retrieval mode comparison ─────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 3: Retrieval Modes")
    print("="*60)
    docs = load_and_chunk(DATA_PATH, strategy="hybrid")
    vs = VectorStore(
        collection_name="exp_retrieval",
        model_key="minilm",
        persist_dir="./chroma_exp/retrieval",
    )
    vs.index(docs)
    for mode in ["semantic", "bm25", "hybrid"]:
        ret = Retriever(vs, docs, mode=mode, use_rewriter=True, use_reranker=False)
        m = quick_eval(ret, k=5)
        row = {"experiment": "retrieval_mode", "variant": mode, "k": 5, **m}
        results.append(row)
        print(f"  {mode:10s} → hit_rate={m['hit_rate']:.3f}  recall={m['recall_k']:.3f}  MRR={m['mrr']:.3f}")

    # ── Exp 4: k values ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 4: k Values (3, 5, 10)")
    print("="*60)
    ret = Retriever(vs, docs, mode="hybrid", use_rewriter=True, use_reranker=False)
    for k_val in [3, 5, 10]:
        m = quick_eval(ret, k=k_val)
        row = {"experiment": "k_value", "variant": str(k_val), "k": k_val, **m}
        results.append(row)
        print(f"  k={k_val} → hit_rate={m['hit_rate']:.3f}  recall={m['recall_k']:.3f}  MRR={m['mrr']:.3f}")

    # ── Exp 5: Query rewriter effect ─────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 5: Query Rewriter")
    print("="*60)
    for use_rw in [False, True]:
        ret = Retriever(vs, docs, mode="hybrid", use_rewriter=use_rw, use_reranker=False)
        m = quick_eval(ret, k=5)
        label = "with_rewriter" if use_rw else "no_rewriter"
        row = {"experiment": "rewriter", "variant": label, "k": 5, **m}
        results.append(row)
        print(f"  {label:16s} → hit_rate={m['hit_rate']:.3f}  recall={m['recall_k']:.3f}  MRR={m['mrr']:.3f}")

    # Save
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n All experiments done. Saved to evaluation/experiment_results.json")
    return results


if __name__ == "__main__":
    run_all_experiments()
