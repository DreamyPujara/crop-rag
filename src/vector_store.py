"""
Embedding + Vector Store
Wraps ChromaDB with sentence-transformers embeddings.
Supports two embedding models for comparison experiments.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import os


EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",       # fast, small
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",       # stronger, slower
}


class VectorStore:
    """
    Wraps ChromaDB persistent store with a chosen sentence-transformer encoder.
    """

    def __init__(
        self,
        collection_name: str = "crop_rag",
        model_key: str = "minilm",
        persist_dir: str = "./chroma_db",
    ):
        self.model_name = EMBEDDING_MODELS[model_key]
        self.model_key = model_key
        print(f"Loading embedding model: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name)

        self.client = chromadb.PersistentClient(path=persist_dir)
        # Delete existing collection to allow re-indexing
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB collection '{collection_name}' ready at '{persist_dir}'")

    def index(self, docs: List[Dict[str, Any]], batch_size: int = 256) -> None:
        """Embed and store all documents in batches."""
        texts = [d["text"] for d in docs]
        ids   = [d["id"]   for d in docs]
        metas = [d["metadata"] for d in docs]

        print(f"Embedding {len(texts)} documents …")
        embeddings = self.encoder.encode(
            texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
        ).tolist()

        # Upsert in batches of 500 (ChromaDB limit)
        for start in range(0, len(docs), 500):
            end = start + 500
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metas[start:end],
            )
        print("Indexing complete.")

    def search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic nearest-neighbour search; returns list of result dicts."""
        q_emb = self.encoder.encode([query], normalize_embeddings=True).tolist()
        kwargs: Dict[str, Any] = {
            "query_embeddings": q_emb,
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter

        results = self.collection.query(**kwargs)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "metadata": meta,
                "score": float(1 - dist),   # cosine similarity
            })
        return hits

    def get_encoder(self):
        return self.encoder
