import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 0 - Title
cells.append(nbf.v4.new_markdown_cell("""# 🌾 Crop Recommendation RAG System
## End-to-End Walkthrough

**Associate AI Engineer Assignment**

This notebook walks through:
1. Dataset exploration
2. Chunking strategy comparison  
3. Vector store indexing with two embedding models
4. Retrieval experiments (semantic vs BM25 vs hybrid)
5. RAG generation with Claude
6. Evaluation with retrieval + LLM-as-judge metrics
"""))

# Cell 1 - Setup
cells.append(nbf.v4.new_code_cell("""import os, sys
sys.path.insert(0, '..')  # run from notebooks/ or root

# Set your API key
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-YOUR_KEY_HERE'

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
print("Setup complete ✓")
"""))

# Cell 2 - Data Exploration
cells.append(nbf.v4.new_markdown_cell("## 1. Dataset Exploration"))

cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('../data/Crop_recommendation.csv')
print(f"Shape: {df.shape}")
print(f"Crops ({df['label'].nunique()}): {sorted(df['label'].unique())}")
df.head()
"""))

cells.append(nbf.v4.new_code_cell("""# Summary statistics per crop
numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
crop_stats = df.groupby('label')[numeric_cols].mean().round(1)
crop_stats
"""))

cells.append(nbf.v4.new_code_cell("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    df.groupby('label')[col].mean().sort_values().plot(
        kind='barh', ax=axes[i], title=f'Mean {col}', color='steelblue')
    axes[i].set_xlabel(col)
axes[-1].set_visible(False)
plt.suptitle('Mean Feature Values per Crop', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../evaluation/feature_distributions.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved to evaluation/feature_distributions.png")
"""))

# Cell 3 - Chunking
cells.append(nbf.v4.new_markdown_cell("""## 2. Chunking Strategy Comparison

Three strategies:
| Strategy | Docs | Pros | Cons |
|----------|------|------|------|
| **Row-level** | 2200 | Granular, exact values | Noisy, redundant |
| **Crop-aggregate** | 22 | Compact, LLM-friendly | Loses outlier info |
| **Hybrid** | 242 | Balance precision + coverage | Slightly larger index |

**Choice: Hybrid** — aggregate docs handle "what are the conditions for X?" while row-level docs help with specific numeric threshold queries.
"""))

cells.append(nbf.v4.new_code_cell("""from src.ingestion import load_and_chunk

for strategy in ['row', 'aggregate', 'hybrid']:
    docs = load_and_chunk('../data/Crop_recommendation.csv', strategy=strategy)
    print(f"{strategy:12s}: {len(docs):5d} docs  |  example: {docs[0]['text'][:80]}...")
"""))

cells.append(nbf.v4.new_code_cell("""# Show one of each type
docs_hybrid = load_and_chunk('../data/Crop_recommendation.csv', strategy='hybrid')

print("=== AGGREGATE DOC (first) ===")
print(docs_hybrid[0]['text'])
print()
print("=== ROW-LEVEL DOC (last) ===")
print(docs_hybrid[-1]['text'])
"""))

# Cell 4 - Embedding
cells.append(nbf.v4.new_markdown_cell("""## 3. Embedding & Indexing

Two models compared:
| Model | Dims | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good |
| `all-mpnet-base-v2` | 768 | Slower | Better |

**Choice for production: MiniLM** — adequate quality with 3× faster indexing and lower memory.
"""))

cells.append(nbf.v4.new_code_cell("""from src.vector_store import VectorStore
import time

docs = load_and_chunk('../data/Crop_recommendation.csv', strategy='hybrid')

t0 = time.time()
vs = VectorStore(collection_name='demo', model_key='minilm', persist_dir='../chroma_demo')
vs.index(docs)
print(f"MiniLM indexed {len(docs)} docs in {time.time()-t0:.1f}s")
"""))

# Cell 5 - Retrieval
cells.append(nbf.v4.new_markdown_cell("## 4. Retrieval Experiments"))

cells.append(nbf.v4.new_code_cell("""from src.retriever import Retriever

query = "What crop grows in high humidity and low nitrogen conditions?"

# Semantic only
ret_sem = Retriever(vs, docs, mode='semantic', use_rewriter=True)
hits_sem = ret_sem.retrieve(query, k=5)

print("=== SEMANTIC RESULTS ===")
for i, h in enumerate(hits_sem, 1):
    print(f"[{i}] score={h['score']:.3f}  crop={h['metadata']['crop']:12s}  {h['text'][:80]}...")
"""))

cells.append(nbf.v4.new_code_cell("""# BM25 only
ret_bm25 = Retriever(vs, docs, mode='bm25', use_rewriter=False)
hits_bm25 = ret_bm25.retrieve(query, k=5)

print("=== BM25 RESULTS ===")
for i, h in enumerate(hits_bm25, 1):
    print(f"[{i}] score={h['score']:.3f}  crop={h['metadata']['crop']:12s}  {h['text'][:80]}...")
"""))

cells.append(nbf.v4.new_code_cell("""# Hybrid
ret_hyb = Retriever(vs, docs, mode='hybrid', use_rewriter=True)
hits_hyb = ret_hyb.retrieve(query, k=5)

print("=== HYBRID RESULTS ===")
for i, h in enumerate(hits_hyb, 1):
    print(f"[{i}] score={h['score']:.5f}  crop={h['metadata']['crop']:12s}  {h['text'][:80]}...")
"""))

# Cell 6 - Query rewriting
cells.append(nbf.v4.new_code_cell("""from src.retriever import rewrite_query

test_queries = [
    "What grows in dry weather?",
    "Crops for humid tropical climate",
    "What needs acidic soil and low N?",
]
for q in test_queries:
    print(f"Original : {q}")
    print(f"Expanded : {rewrite_query(q)}")
    print()
"""))

# Cell 7 - Generation
cells.append(nbf.v4.new_markdown_cell("## 5. Generation with Claude"))

cells.append(nbf.v4.new_code_cell("""from src.generator import generate_answer

query = "Which crops are suitable for a humid tropical climate?"
hits = ret_hyb.retrieve(query, k=5)

result = generate_answer(query, hits, api_key=os.environ['ANTHROPIC_API_KEY'])

print("ANSWER:")
print(result['answer'])
print(f"\\nTokens: in={result['input_tokens']}  out={result['output_tokens']}")
"""))

cells.append(nbf.v4.new_code_cell("""# Test a comparison query
query2 = "Compare nitrogen requirements for rice vs wheat."
hits2 = ret_hyb.retrieve(query2, k=5)
result2 = generate_answer(query2, hits2, api_key=os.environ['ANTHROPIC_API_KEY'])
print(result2['answer'])
"""))

cells.append(nbf.v4.new_code_cell("""# Test with metadata filter — only search rice and wheat
query3 = "What are the pH preferences for rice?"
hits3 = vs.search(query3, k=5, metadata_filter={"crop": "rice"})
result3 = generate_answer(query3, hits3, api_key=os.environ['ANTHROPIC_API_KEY'])
print(result3['answer'])
"""))

# Cell 8 - Evaluation
cells.append(nbf.v4.new_markdown_cell("## 6. Evaluation"))

cells.append(nbf.v4.new_code_cell("""from evaluation.evaluator import (
    TEST_QA_PAIRS, compute_retrieval_metrics,
    extract_crops_from_hits, llm_judge
)

# Quick retrieval eval on first 10 questions
ret_metrics_all = []
for qa in TEST_QA_PAIRS[:10]:
    hits = ret_hyb.retrieve(qa['question'], k=5)
    crops = extract_crops_from_hits(hits)
    m = compute_retrieval_metrics(crops, qa['expected_crops'], k=5)
    ret_metrics_all.append({**qa, **m, 'retrieved_crops': crops})

import pandas as pd
results_df = pd.DataFrame(ret_metrics_all)[
    ['id', 'question', 'expected_crops', 'retrieved_crops',
     'hit_rate', 'precision_k', 'recall_k', 'rr']
]
print(f"\\nMean Hit Rate : {results_df.hit_rate.mean():.3f}")
print(f"Mean Recall@5 : {results_df.recall_k.mean():.3f}")
print(f"Mean MRR      : {results_df.rr.mean():.3f}")
results_df[['id','hit_rate','recall_k','rr']].head(10)
"""))

cells.append(nbf.v4.new_code_cell("""# LLM-as-judge on 3 samples
print("Running LLM judge on 3 sample Q&A pairs...")
judge_results = []
for qa in TEST_QA_PAIRS[:3]:
    hits = ret_hyb.retrieve(qa['question'], k=5)
    gen = generate_answer(qa['question'], hits, api_key=os.environ['ANTHROPIC_API_KEY'])
    context = " ".join(h['text'] for h in hits)
    scores = llm_judge(qa['question'], qa['ground_truth'], gen['answer'], context,
                       api_key=os.environ['ANTHROPIC_API_KEY'])
    judge_results.append({'question': qa['question'][:60], **scores})
    print(f"  {qa['id']}: faith={scores.get('faithfulness',0):.2f}  "
          f"relevance={scores.get('relevance',0):.2f}  "
          f"correctness={scores.get('correctness',0):.2f}  "
          f"hallucination={scores.get('hallucination',0):.2f}")

pd.DataFrame(judge_results)
"""))

# Cell 9 - Experiment summary
cells.append(nbf.v4.new_markdown_cell("""## 7. Experiment Summary

Run all experiments with:
```python
from evaluation.experiments import run_all_experiments
results = run_all_experiments()
```

Or view the saved results:
"""))

cells.append(nbf.v4.new_code_cell("""import json, os

results_path = '../evaluation/experiment_results.json'
if os.path.exists(results_path):
    with open(results_path) as f:
        exp_data = json.load(f)
    exp_df = pd.DataFrame(exp_data)
    
    for exp_name, grp in exp_df.groupby('experiment'):
        print(f"\\n── {exp_name} ──")
        print(grp[['variant','hit_rate','recall_k','mrr']].to_string(index=False))
else:
    print("Run evaluation/experiments.py first to generate experiment results.")
    print("Command: cd .. && python -m evaluation.experiments")
"""))

# Cell 10 - Full pipeline
cells.append(nbf.v4.new_markdown_cell("## 8. Full Pipeline Demo"))

cells.append(nbf.v4.new_code_cell("""from src.pipeline import CropRAGPipeline

rag = CropRAGPipeline(
    csv_path='../data/Crop_recommendation.csv',
    chunk_strategy='hybrid',
    embedding_model='minilm',
    retrieval_mode='hybrid',
    use_rewriter=True,
    k=5,
    api_key=os.environ['ANTHROPIC_API_KEY'],
    collection_name='full_demo',
    persist_dir='../chroma_full',
)
"""))

cells.append(nbf.v4.new_code_cell("""questions = [
    "What crop grows best with high nitrogen, low rainfall, and acidic soil?",
    "Which crops are suitable for a humid tropical climate?",
    "What should I grow if my soil pH is 5.5 and rainfall is 250mm?",
]

for q in questions:
    result = rag.query(q, verbose=False)
    print(f"Q: {q}")
    print(f"A: {result['answer'][:300]}...")
    print()
"""))

nb.cells = cells
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.10.0'
    }
}

import nbformat
with open('../notebooks/crop_rag_demo.ipynb', 'w') as f:
    nbformat.write(nb, f)
print("Notebook created: notebooks/crop_rag_demo.ipynb")
