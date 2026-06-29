import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# 🌾 Crop Recommendation RAG System — End-to-End Notebook

**Assignment: Associate AI Engineer | RAG System**

This notebook walks through the complete pipeline:
1. Dataset exploration & statistics
2. Chunking strategy comparison (row vs aggregate vs hybrid)
3. Vector indexing with two embedding models
4. Retrieval experiments (Semantic / BM25 / Hybrid + RRF)
5. Query rewriting for agronomic shorthand
6. Grounded answer generation with Claude
7. Evaluation: Retrieval metrics + LLM-as-judge
"""))

cells.append(nbf.v4.new_code_cell("""import os, sys, warnings
sys.path.insert(0, '..')   # adjust if running from repo root
warnings.filterwarnings('ignore')

# ⚠️  Set your Anthropic API key
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-YOUR_KEY_HERE'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print('Setup complete ✓')
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Dataset Exploration"))

cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('../data/Crop_recommendation.csv')
print(f'Shape: {df.shape}')
print(f'Crops ({df[\"label\"].nunique()}): {sorted(df[\"label\"].unique())}')
df.describe().round(2)
"""))

cells.append(nbf.v4.new_code_cell("""numeric_cols = ['N','P','K','temperature','humidity','ph','rainfall']

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    df.groupby('label')[col].mean().sort_values().plot(
        kind='barh', ax=axes[i], title=f'Mean {col}', color='steelblue', edgecolor='white')
axes[-1].set_visible(False)
plt.suptitle('Mean Feature Values per Crop', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../evaluation/feature_distributions.png', dpi=100, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Chunking Strategies

| Strategy | # Docs | Description |
|----------|--------|-------------|
| **row** | 2200 | One sentence per CSV row |
| **aggregate** | 22 | One summary paragraph per crop (mean + range) |
| **hybrid** | 242 | Aggregate (22) + 10 sampled rows per crop |

**Decision: Hybrid** — aggregate docs answer "what conditions does crop X need?" while row docs provide granular retrieval for threshold-based queries like "pH < 5.5 and rainfall < 100mm".
"""))

cells.append(nbf.v4.new_code_cell("""from src.ingestion import load_and_chunk

for strategy in ['row', 'aggregate', 'hybrid']:
    docs = load_and_chunk('../data/Crop_recommendation.csv', strategy=strategy)
    print(f'{strategy:12s}: {len(docs):5d} docs')
    print(f'  Example: {docs[0][\"text\"][:100]}...')
    print()
"""))

cells.append(nbf.v4.new_code_cell("""docs = load_and_chunk('../data/Crop_recommendation.csv', strategy='hybrid')
print('=== AGGREGATE DOC ===')
print(docs[0]['text'])
print()
print('=== ROW-LEVEL DOC ===')
print(docs[-1]['text'])
"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Embedding Models

| Model | Dims | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` | 384 | ~3× faster | Good baseline |
| `all-mpnet-base-v2` | 768 | Slower | ~5% better recall |

**Production choice: MiniLM** — adequate quality, 3× faster indexing, lower RAM.
"""))

cells.append(nbf.v4.new_code_cell("""import time
from src.vector_store import VectorStore

docs = load_and_chunk('../data/Crop_recommendation.csv', strategy='hybrid')

t0 = time.time()
vs = VectorStore(collection_name='notebook_demo', model_key='minilm',
                 persist_dir='../chroma_notebook')
vs.index(docs)
elapsed = time.time() - t0
print(f'MiniLM indexed {len(docs)} docs in {elapsed:.1f}s')
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Retrieval Experiments"))

cells.append(nbf.v4.new_code_cell("""from src.retriever import Retriever, rewrite_query

# Show query rewriting
test_queries = [
    'What grows in dry weather?',
    'Crops for humid tropical climate',
    'I need something for acidic soil and low N',
]
print('=== Query Rewriting ===')
for q in test_queries:
    print(f'  Before: {q}')
    print(f'  After : {rewrite_query(q)}')
    print()
"""))

cells.append(nbf.v4.new_code_cell("""query = 'What crop grows in high humidity and very low rainfall?'

ret_sem  = Retriever(vs, docs, mode='semantic', use_rewriter=True)
ret_bm25 = Retriever(vs, docs, mode='bm25',     use_rewriter=False)
ret_hyb  = Retriever(vs, docs, mode='hybrid',   use_rewriter=True)

for label, ret in [('Semantic', ret_sem), ('BM25', ret_bm25), ('Hybrid', ret_hyb)]:
    hits = ret.retrieve(query, k=5)
    crops = [h['metadata']['crop'] for h in hits]
    print(f'{label:10s}: {crops}')
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Generation with Claude"))

cells.append(nbf.v4.new_code_cell("""from src.generator import generate_answer

questions = [
    'What crop grows best with high nitrogen, low rainfall, and acidic soil?',
    'Which crops are suitable for a humid tropical climate?',
    'Compare nitrogen requirements for rice vs wheat.',
    'What should I grow if my soil pH is 5.5 and rainfall is 250mm?',
]

for q in questions:
    hits = ret_hyb.retrieve(q, k=5)
    result = generate_answer(q, hits, api_key=os.environ['ANTHROPIC_API_KEY'])
    print(f'Q: {q}')
    print(f'A: {result[\"answer\"][:400]}')
    print(f'   [tokens: in={result[\"input_tokens\"]} out={result[\"output_tokens\"]}]')
    print()
"""))

cells.append(nbf.v4.new_code_cell("""# Metadata-filtered retrieval
q = 'What pH does rice prefer?'
hits_rice = vs.search(q, k=5, metadata_filter={'crop': 'rice'})
result = generate_answer(q, hits_rice, api_key=os.environ['ANTHROPIC_API_KEY'])
print('=== With metadata filter: crop=rice ===')
print(result['answer'])
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Evaluation"))

cells.append(nbf.v4.new_code_cell("""from evaluation.evaluator import (
    TEST_QA_PAIRS, compute_retrieval_metrics,
    extract_crops_from_hits, llm_judge
)

print(f'Test set: {len(TEST_QA_PAIRS)} Q&A pairs')
print('Sample:')
for qa in TEST_QA_PAIRS[:3]:
    print(f'  [{qa[\"id\"]}] {qa[\"question\"]}')
    print(f'         Expected crops: {qa[\"expected_crops\"]}')
    print()
"""))

cells.append(nbf.v4.new_code_cell("""# Retrieval metrics — first 15 questions
ret_metrics_all = []
for qa in TEST_QA_PAIRS[:15]:
    hits = ret_hyb.retrieve(qa['question'], k=5)
    crops = extract_crops_from_hits(hits)
    m = compute_retrieval_metrics(crops, qa['expected_crops'], k=5)
    ret_metrics_all.append({'id': qa['id'], **m, 'retrieved': crops[:3]})

res_df = pd.DataFrame(ret_metrics_all)
print(f'Mean Hit Rate  @ 5: {res_df.hit_rate.mean():.3f}')
print(f'Mean Precision@ 5: {res_df.precision_k.mean():.3f}')
print(f'Mean Recall   @ 5: {res_df.recall_k.mean():.3f}')
print(f'Mean MRR         : {res_df.rr.mean():.3f}')
res_df[['id','hit_rate','recall_k','rr']].head(15)
"""))

cells.append(nbf.v4.new_code_cell("""# LLM-as-Judge on 3 pairs
judge_rows = []
for qa in TEST_QA_PAIRS[:3]:
    hits = ret_hyb.retrieve(qa['question'], k=5)
    gen = generate_answer(qa['question'], hits, api_key=os.environ['ANTHROPIC_API_KEY'])
    ctx = ' '.join(h['text'] for h in hits)
    scores = llm_judge(qa['question'], qa['ground_truth'], gen['answer'], ctx,
                       api_key=os.environ['ANTHROPIC_API_KEY'])
    judge_rows.append({'id': qa['id'], **scores})
    print(f'{qa[\"id\"]}: {scores}')

pd.DataFrame(judge_rows)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 7. Full Evaluation Run

To run the complete 27-question evaluation with LLM judge:

```python
from evaluation.evaluator import run_evaluation
from src.pipeline import CropRAGPipeline

rag = CropRAGPipeline(
    csv_path='../data/Crop_recommendation.csv',
    chunk_strategy='hybrid',
    embedding_model='minilm',
    retrieval_mode='hybrid',
    api_key=os.environ['ANTHROPIC_API_KEY'],
)

output = run_evaluation(rag, k=5, n_questions=27, use_llm_judge=True)
```

Or via CLI:
```bash
python cli.py --evaluate --api-key $ANTHROPIC_API_KEY
```
"""))

cells.append(nbf.v4.new_markdown_cell("""## Architecture Summary

```
CSV Dataset (2200 rows)
        │
        ▼
┌──────────────────────────────┐
│  Ingestion Pipeline           │
│  - load_and_chunk()           │
│  - Strategy: hybrid (242 docs)│
└──────────────┬───────────────┘
               │
        ┌──────▼──────┐
        │  Embeddings  │
        │  MiniLM-L6   │
        └──────┬───────┘
               │
        ┌──────▼──────┐
        │  ChromaDB    │
        │  (cosine)    │
        └──────┬───────┘
               │
     ┌─────────▼──────────┐
     │  Retriever          │
     │  BM25 + Semantic    │
     │  → RRF Fusion       │
     │  → Query Rewriter   │
     └─────────┬───────────┘
               │  top-k chunks
     ┌─────────▼───────────┐
     │  Generator (Claude)  │
     │  Grounded + Cited    │
     └─────────────────────┘
```
"""))

nb.cells = cells
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10.0'}
}

with open('../notebooks/crop_rag_demo.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook saved: notebooks/crop_rag_demo.ipynb")
