#!/usr/bin/env python3
"""
CLI Demo — Crop Recommendation RAG System
Usage:
    python cli.py --query "What crop grows best with high nitrogen and acidic soil?"
    python cli.py --interactive
    python cli.py --evaluate
"""

import argparse
import os
import sys

from src.pipeline import CropRAGPipeline


SAMPLE_QUERIES = [
    "What crop grows best with high nitrogen, low rainfall, and acidic soil?",
    "Which crops are suitable for a humid tropical climate?",
    "Compare nitrogen requirements for rice vs wheat.",
    "What should I grow if my soil pH is 5.5 and rainfall is 250mm?",
    "Which crops need the lowest water?",
]


def build_pipeline(args) -> CropRAGPipeline:
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY env var or pass --api-key")
        sys.exit(1)

    return CropRAGPipeline(
        csv_path=args.data,
        chunk_strategy=args.chunk_strategy,
        embedding_model=args.embedding_model,
        retrieval_mode=args.retrieval_mode,
        use_rewriter=not args.no_rewriter,
        use_reranker=args.reranker,
        k=args.k,
        api_key=api_key,
    )


def run_single_query(pipeline: CropRAGPipeline, query: str, verbose: bool = False):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    result = pipeline.query(query, verbose=verbose)
    print(f"\n{result['answer']}")
    if verbose:
        print(f"\n[tokens: in={result['input_tokens']} out={result['output_tokens']}]")


def interactive_mode(pipeline: CropRAGPipeline):
    print("\n🌾 Crop Recommendation RAG  (type 'quit' to exit)\n")
    while True:
        try:
            query = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        run_single_query(pipeline, query, verbose=True)


def main():
    parser = argparse.ArgumentParser(description="Crop Recommendation RAG CLI")
    parser.add_argument("--query",          type=str,   help="Single question to answer")
    parser.add_argument("--interactive",    action="store_true", help="Interactive Q&A mode")
    parser.add_argument("--evaluate",       action="store_true", help="Run evaluation suite")
    parser.add_argument("--demo",           action="store_true", help="Run 5 sample queries")
    parser.add_argument("--data",           default="data/Crop_recommendation.csv")
    parser.add_argument("--chunk-strategy", default="hybrid",
                        choices=["row", "aggregate", "hybrid"])
    parser.add_argument("--embedding-model",default="minilm",
                        choices=["minilm", "mpnet"])
    parser.add_argument("--retrieval-mode", default="hybrid",
                        choices=["semantic", "bm25", "hybrid"])
    parser.add_argument("--k",              type=int,   default=5)
    parser.add_argument("--api-key",        type=str,   default=None)
    parser.add_argument("--reranker",       action="store_true")
    parser.add_argument("--no-rewriter",    action="store_true")
    parser.add_argument("--verbose",        action="store_true")
    parser.add_argument("--eval-n",         type=int,   default=27)
    parser.add_argument("--no-llm-judge",   action="store_true")
    args = parser.parse_args()

    pipeline = build_pipeline(args)

    if args.query:
        run_single_query(pipeline, args.query, verbose=args.verbose)

    elif args.interactive:
        interactive_mode(pipeline)

    elif args.demo:
        for q in SAMPLE_QUERIES:
            run_single_query(pipeline, q, verbose=args.verbose)

    elif args.evaluate:
        from evaluation.evaluator import run_evaluation
        run_evaluation(
            pipeline,
            k=args.k,
            n_questions=args.eval_n,
            use_llm_judge=not args.no_llm_judge,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
