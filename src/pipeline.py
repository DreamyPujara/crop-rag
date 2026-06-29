"""
RAG Pipeline — end-to-end orchestrator.
Usage:
    from src.pipeline import CropRAGPipeline
    rag = CropRAGPipeline(api_key="sk-ant-...")
    result = rag.query("What crop grows best with high nitrogen and acidic soil?")
    print(result["answer"])
"""

from __future__ import annotations
import os
from typing import Optional, Literal

from src.ingestion import load_and_chunk
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import generate_answer


class CropRAGPipeline:
    """
    One-stop class that wires ingestion → indexing → retrieval → generation.

    Parameters
    ----------
    csv_path       : path to Crop_recommendation.csv
    chunk_strategy : "row" | "aggregate" | "hybrid"
    embedding_model: "minilm" | "mpnet"
    retrieval_mode : "semantic" | "bm25" | "hybrid"
    use_rewriter   : expand agronomic shorthand before retrieval
    use_reranker   : CrossEncoder reranker (slower, better precision)
    k              : number of chunks to retrieve
    persist_dir    : directory for ChromaDB persistence
    api_key        : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
    """

    def __init__(
        self,
        csv_path: str = "data/Crop_recommendation.csv",
        chunk_strategy: Literal["row", "aggregate", "hybrid"] = "hybrid",
        embedding_model: str = "minilm",
        retrieval_mode: str = "hybrid",
        use_rewriter: bool = True,
        use_reranker: bool = False,
        k: int = 5,
        persist_dir: str = "./chroma_db",
        api_key: Optional[str] = None,
        collection_name: str = "crop_rag",
    ):
        self.k = k
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        # 1. Ingest & chunk
        print(f"\n[1/3] Ingestion — strategy='{chunk_strategy}'")
        self.docs = load_and_chunk(csv_path, strategy=chunk_strategy)

        # 2. Embed & index
        print(f"\n[2/3] Embedding & indexing — model='{embedding_model}'")
        self.vs = VectorStore(
            collection_name=collection_name,
            model_key=embedding_model,
            persist_dir=persist_dir,
        )
        self.vs.index(self.docs)

        # 3. Build retriever
        print(f"\n[3/3] Building retriever — mode='{retrieval_mode}'")
        self.retriever = Retriever(
            vector_store=self.vs,
            docs=self.docs,
            mode=retrieval_mode,
            use_rewriter=use_rewriter,
            use_reranker=use_reranker,
        )
        print("\n✅ Pipeline ready.\n")

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        """Run a full RAG query and return the answer dict."""
        hits = self.retriever.retrieve(
            question, k=k or self.k, metadata_filter=metadata_filter
        )
        if verbose:
            print(f"\n--- Retrieved {len(hits)} chunks ---")
            for i, h in enumerate(hits, 1):
                print(f"[{i}] score={h['score']:.3f}  {h['text'][:120]}")
            print()

        result = generate_answer(question, hits, api_key=self.api_key)
        return result
