"""
Evaluation Module
- 27 Q&A ground-truth pairs derived from dataset
- Retrieval metrics: Recall@K, Precision@K, MRR, Hit Rate
- Generation metrics: Faithfulness, Relevance, Hallucination rate (LLM-as-judge)
"""

from __future__ import annotations
import json
import os
import re
import time
from typing import List, Dict, Any, Optional
import numpy as np
import anthropic


# ── Ground-truth test set (27 Q&A pairs) ─────────────────────────────────────

TEST_QA_PAIRS = [
    # Numeric-condition queries
    {
        "id": "q01",
        "question": "What crop grows best with high nitrogen (~90 kg/ha), low rainfall (~200mm), and neutral pH (~6.5)?",
        "expected_crops": ["rice"],
        "ground_truth": "Rice is recommended with nitrogen around 90 kg/ha, rainfall ~200mm, and pH ~6.5.",
    },
    {
        "id": "q02",
        "question": "Which crop is suitable for very high potassium (>200 kg/ha) and high humidity?",
        "expected_crops": ["coconut"],
        "ground_truth": "Coconut requires high potassium levels and grows in humid conditions.",
    },
    {
        "id": "q03",
        "question": "What should I grow if my soil pH is 5.5 and rainfall is 250mm?",
        "expected_crops": ["rice", "maize", "coffee"],
        "ground_truth": "Crops like rice, maize, or coffee can grow in slightly acidic soil with moderate rainfall.",
    },
    {
        "id": "q04",
        "question": "Which crop needs the highest phosphorus content?",
        "expected_crops": ["grapes"],
        "ground_truth": "Grapes require high phosphorus levels in the soil.",
    },
    {
        "id": "q05",
        "question": "What crop grows in temperatures above 35°C with moderate rainfall?",
        "expected_crops": ["jute", "cotton"],
        "ground_truth": "Jute and cotton can tolerate high temperatures above 35°C.",
    },
    # Comparative queries
    {
        "id": "q06",
        "question": "Compare nitrogen requirements for rice vs wheat.",
        "expected_crops": ["rice", "wheat"],
        "ground_truth": "Rice requires higher nitrogen (~80-100 kg/ha) compared to wheat which needs moderate nitrogen.",
    },
    {
        "id": "q07",
        "question": "Which has higher potassium requirements — banana or mango?",
        "expected_crops": ["banana", "mango"],
        "ground_truth": "Banana generally requires higher potassium than mango.",
    },
    {
        "id": "q08",
        "question": "Compare rainfall requirements for cotton vs rice.",
        "expected_crops": ["cotton", "rice"],
        "ground_truth": "Rice requires significantly more rainfall than cotton.",
    },
    {
        "id": "q09",
        "question": "Which crop tolerates lower pH — coffee or lentil?",
        "expected_crops": ["coffee", "lentil"],
        "ground_truth": "Coffee tolerates lower (more acidic) pH than lentil.",
    },
    {
        "id": "q10",
        "question": "Compare temperature ranges for watermelon vs apple.",
        "expected_crops": ["watermelon", "apple"],
        "ground_truth": "Watermelon prefers higher temperatures while apple prefers cooler temperatures.",
    },
    # Climate/condition queries
    {
        "id": "q11",
        "question": "Which crops are suitable for a humid tropical climate?",
        "expected_crops": ["rice", "coconut", "banana", "papaya", "jute"],
        "ground_truth": "Rice, coconut, banana, papaya, and jute thrive in humid tropical climates.",
    },
    {
        "id": "q12",
        "question": "What crops grow well in dry conditions with low rainfall?",
        "expected_crops": ["chickpea", "mothbeans", "cotton"],
        "ground_truth": "Chickpea, mothbeans, and cotton are drought-tolerant crops.",
    },
    {
        "id": "q13",
        "question": "Which crops prefer acidic soil (pH below 6)?",
        "expected_crops": ["coffee", "blueberry", "rice"],
        "ground_truth": "Coffee grows best in acidic soil with pH below 6.",
    },
    {
        "id": "q14",
        "question": "What crop is best for alkaline soil with pH above 8?",
        "expected_crops": ["chickpea", "kidneybeans"],
        "ground_truth": "Certain legumes like chickpea can tolerate alkaline soils.",
    },
    {
        "id": "q15",
        "question": "Which crops need very high humidity (above 80%)?",
        "expected_crops": ["rice", "coconut", "banana"],
        "ground_truth": "Rice, coconut, and banana require high humidity above 80%.",
    },
    # Specific crop profiles
    {
        "id": "q16",
        "question": "What are the ideal growing conditions for coffee?",
        "expected_crops": ["coffee"],
        "ground_truth": "Coffee prefers moderate temperature, high rainfall, acidic soil (pH 5-6), and moderate NPK.",
    },
    {
        "id": "q17",
        "question": "What NPK ratio does maize need?",
        "expected_crops": ["maize"],
        "ground_truth": "Maize requires balanced NPK with moderate nitrogen, phosphorus, and potassium.",
    },
    {
        "id": "q18",
        "question": "Describe the ideal conditions for growing pomegranate.",
        "expected_crops": ["pomegranate"],
        "ground_truth": "Pomegranate prefers low humidity, moderate temperature, and low-to-moderate rainfall.",
    },
    {
        "id": "q19",
        "question": "What soil conditions does cotton need?",
        "expected_crops": ["cotton"],
        "ground_truth": "Cotton needs moderate NPK, high temperature, moderate rainfall, and neutral pH.",
    },
    {
        "id": "q20",
        "question": "What are the nitrogen and potassium needs for banana?",
        "expected_crops": ["banana"],
        "ground_truth": "Banana requires high nitrogen and high potassium levels.",
    },
    # Recommendation queries
    {
        "id": "q21",
        "question": "What crop should I plant if my soil has N=20, P=30, K=20, pH=7, rainfall=80mm, temperature=28°C?",
        "expected_crops": ["chickpea", "mothbeans", "pigeonpeas"],
        "ground_truth": "With low nutrient levels and low rainfall, legumes like chickpea or mothbeans are suitable.",
    },
    {
        "id": "q22",
        "question": "I have high humidity (90%) and temperature of 30°C. What should I grow?",
        "expected_crops": ["rice", "coconut", "jute"],
        "ground_truth": "High humidity and warm temperature is ideal for rice, coconut, or jute.",
    },
    {
        "id": "q23",
        "question": "What can I grow in cool temperatures around 15°C?",
        "expected_crops": ["apple", "grapes"],
        "ground_truth": "Apple and grapes prefer cooler temperatures around 15-20°C.",
    },
    {
        "id": "q24",
        "question": "Which legumes grow in low-rainfall conditions?",
        "expected_crops": ["chickpea", "lentil", "mothbeans", "pigeonpeas"],
        "ground_truth": "Chickpea, lentil, mothbeans, and pigeonpeas are drought-tolerant legumes.",
    },
    {
        "id": "q25",
        "question": "What fruit crop grows well with high potassium and low rainfall?",
        "expected_crops": ["pomegranate", "muskmelon"],
        "ground_truth": "Pomegranate and muskmelon can thrive with high potassium and low rainfall.",
    },
    {
        "id": "q26",
        "question": "Which crops are best for waterlogged or very high rainfall areas (>250mm)?",
        "expected_crops": ["rice", "coconut", "jute"],
        "ground_truth": "Rice, coconut, and jute are suited for high-rainfall waterlogged conditions.",
    },
    {
        "id": "q27",
        "question": "What is the average rainfall requirement for mango?",
        "expected_crops": ["mango"],
        "ground_truth": "Mango requires moderate rainfall, typically around 100-150mm annually.",
    },
]


# ── Retrieval Metrics ─────────────────────────────────────────────────────────

def compute_retrieval_metrics(
    retrieved_crops: List[str],
    expected_crops: List[str],
    k: int,
) -> Dict[str, float]:
    """Compute Precision@K, Recall@K, Hit Rate, and reciprocal rank."""
    retrieved_set = set(c.lower() for c in retrieved_crops)
    expected_set  = set(c.lower() for c in expected_crops)

    hits = retrieved_set & expected_set
    precision_k = len(hits) / k if k > 0 else 0.0
    recall_k    = len(hits) / len(expected_set) if expected_set else 0.0
    hit_rate    = 1.0 if hits else 0.0

    # MRR — rank of first relevant hit
    rr = 0.0
    for rank, crop in enumerate(retrieved_crops, 1):
        if crop.lower() in expected_set:
            rr = 1.0 / rank
            break

    return {
        "precision_k": precision_k,
        "recall_k": recall_k,
        "hit_rate": hit_rate,
        "rr": rr,
    }


def extract_crops_from_hits(hits: List[Dict[str, Any]]) -> List[str]:
    """Pull crop labels from retrieved hit metadata."""
    seen, result = set(), []
    for h in hits:
        crop = h.get("metadata", {}).get("crop", "")
        if crop and crop not in seen:
            seen.add(crop)
            result.append(crop)
    return result


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an evaluation judge. Score the following RAG answer on three criteria.

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {answer}
Retrieved Context: {context}

Score each criterion from 0.0 to 1.0 and output ONLY a JSON object:
{{
  "faithfulness": <0-1, is every claim grounded in the context?>,
  "relevance": <0-1, does the answer address the question?>,
  "correctness": <0-1, how closely does it match the ground truth?>,
  "hallucination": <0-1, fraction of answer that is hallucinated (0=none, 1=all)>
}}
Output only the JSON, no other text."""


def llm_judge(
    question: str,
    ground_truth: str,
    answer: str,
    context: str,
    api_key: Optional[str] = None,
) -> Dict[str, float]:
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        context=context[:2000],
    )
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        scores = json.loads(raw)
        return {k: float(v) for k, v in scores.items()}
    except Exception as e:
        print(f"  Judge error: {e}")
        return {"faithfulness": 0.0, "relevance": 0.0, "correctness": 0.0, "hallucination": 1.0}


# ── Full evaluation run ───────────────────────────────────────────────────────

def run_evaluation(
    pipeline,
    k: int = 5,
    n_questions: int = 27,
    output_path: str = "evaluation/eval_results.json",
    use_llm_judge: bool = True,
    sleep_between: float = 1.0,
) -> Dict[str, Any]:
    """
    Run full evaluation on the test set.
    Returns aggregated metrics dict and saves raw results to JSON.
    """
    results = []
    api_key = pipeline.api_key

    qs = TEST_QA_PAIRS[:n_questions]
    print(f"\n{'='*60}")
    print(f"Evaluating {len(qs)} questions  (k={k})")
    print(f"{'='*60}\n")

    for i, qa in enumerate(qs, 1):
        print(f"[{i}/{len(qs)}] {qa['id']}: {qa['question'][:70]}…")

        # Retrieve
        hits = pipeline.retriever.retrieve(qa["question"], k=k)
        retrieved_crops = extract_crops_from_hits(hits)

        # Retrieval metrics
        ret_metrics = compute_retrieval_metrics(
            retrieved_crops, qa["expected_crops"], k=k
        )

        # Generate answer
        from src.generator import generate_answer
        gen_result = generate_answer(qa["question"], hits, api_key=api_key)
        answer = gen_result["answer"]

        # LLM judge
        gen_metrics = {}
        if use_llm_judge:
            context_str = "\n".join(h["text"] for h in hits)
            gen_metrics = llm_judge(
                qa["question"], qa["ground_truth"], answer, context_str, api_key
            )
            time.sleep(sleep_between)

        row = {
            "id": qa["id"],
            "question": qa["question"],
            "expected_crops": qa["expected_crops"],
            "retrieved_crops": retrieved_crops,
            "answer": answer,
            **ret_metrics,
            **gen_metrics,
        }
        results.append(row)

        print(
            f"  hit_rate={ret_metrics['hit_rate']:.2f}  "
            f"recall={ret_metrics['recall_k']:.2f}  "
            f"MRR={ret_metrics['rr']:.2f}"
            + (f"  faith={gen_metrics.get('faithfulness',0):.2f}" if gen_metrics else "")
        )

    # Aggregate
    agg = {
        "k": k,
        "n_questions": len(results),
        "precision_at_k":  np.mean([r["precision_k"]  for r in results]),
        "recall_at_k":     np.mean([r["recall_k"]     for r in results]),
        "hit_rate":        np.mean([r["hit_rate"]      for r in results]),
        "mrr":             np.mean([r["rr"]            for r in results]),
    }
    if use_llm_judge:
        agg.update({
            "faithfulness":  np.mean([r.get("faithfulness", 0)  for r in results]),
            "relevance":     np.mean([r.get("relevance", 0)     for r in results]),
            "correctness":   np.mean([r.get("correctness", 0)   for r in results]),
            "hallucination": np.mean([r.get("hallucination", 0) for r in results]),
        })

    output = {"aggregated": agg, "per_question": results}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n Evaluation complete. Results saved to {output_path}")
    print("\n── Aggregated Metrics ──")
    for k2, v in agg.items():
        if isinstance(v, float):
            print(f"  {k2:20s}: {v:.4f}")

    return output
