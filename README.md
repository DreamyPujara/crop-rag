# 🌾 Crop Recommendation RAG System

A production-style Retrieval-Augmented Generation (RAG) system that answers natural-language queries about optimal crop recommendations, grounded strictly in the [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).

## Architecture

```
CSV (2200 rows, 22 crops)
    → Hybrid chunking (242 docs)
    → MiniLM embeddings
    → ChromaDB vector store
    → BM25 + Semantic hybrid retrieval (RRF)
    → Query rewriting
    → Claude Sonnet 4.6 generation (grounded + cited)
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/crop-rag.git
cd crop-rag

pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...

# Run 5 demo queries
python cli.py --demo

# Interactive Q&A
python cli.py --interactive

# Single query
python cli.py --query "What crop grows best with high nitrogen and acidic soil?"

# Full evaluation (27 Q&A pairs + LLM judge)
python cli.py --evaluate
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-strategy` | `hybrid` | `row` / `aggregate` / `hybrid` |
| `--embedding-model` | `minilm` | `minilm` / `mpnet` |
| `--retrieval-mode` | `hybrid` | `semantic` / `bm25` / `hybrid` |
| `--k` | `5` | Number of chunks to retrieve |
| `--reranker` | off | Enable CrossEncoder reranker |
| `--no-rewriter` | off | Disable query rewriting |
| `--verbose` | off | Show retrieved chunks |

## Project Structure

```
crop-rag/
├── data/
│   └── Crop_recommendation.csv
├── src/
│   ├── ingestion.py      # CSV → documents (3 strategies)
│   ├── vector_store.py   # ChromaDB + SentenceTransformers
│   ├── retriever.py      # BM25, Semantic, Hybrid + RRF, rewriter, reranker
│   ├── generator.py      # Claude generation with citation grounding
│   └── pipeline.py       # End-to-end orchestrator
├── evaluation/
│   ├── evaluator.py      # 27 Q&A pairs + metrics + LLM judge
│   └── experiments.py    # Systematic experiment runner
├── notebooks/
│   └── crop_rag_demo.ipynb
├── cli.py
├── requirements.txt
└── EVALUATION_REPORT.md
```

## Example Queries

```
"What crop grows best with high nitrogen, low rainfall, and acidic soil?"
"Which crops are suitable for a humid tropical climate?"
"Compare nitrogen requirements for rice vs wheat."
"What should I grow if my soil pH is 5.5 and rainfall is 250mm?"
"Which legumes need the least water?"
```

## Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Chunking | Hybrid | Aggregate + sampled rows for best coverage |
| Embeddings | MiniLM | 3× faster than mpnet, adequate recall |
| Vector DB | ChromaDB | Local, persistent, no infra needed |
| Retrieval | BM25 + Semantic (RRF) | +5pp hit rate over semantic-only |
| LLM | Claude Sonnet 4.6 | Strong grounding, citation-aware |

## Evaluation Results (k=5, 27 questions)

| Metric | Score |
|--------|-------|
| Hit Rate | 0.85 |
| Recall@5 | 0.72 |
| MRR | 0.78 |
| Faithfulness | 0.91 |
| Hallucination rate | 0.07 |

See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for full experiment comparisons and failure analysis.
