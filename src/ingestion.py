"""
Ingestion Pipeline for Crop Recommendation RAG System
Supports row-level, crop-aggregate, and hybrid chunking strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, List, Dict, Any


# ── Chunking strategies ──────────────────────────────────────────────────────

def row_to_text(row: pd.Series) -> str:
    """Convert a single CSV row into a natural-language sentence."""
    return (
        f"{row['label'].capitalize()} grows well with "
        f"N={row['N']:.0f} kg/ha, P={row['P']:.0f} kg/ha, K={row['K']:.0f} kg/ha, "
        f"temperature={row['temperature']:.1f}°C, humidity={row['humidity']:.1f}%, "
        f"pH={row['ph']:.2f}, rainfall={row['rainfall']:.1f} mm."
    )


def build_row_level_docs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Strategy 1: One document per CSV row."""
    docs = []
    for idx, row in df.iterrows():
        docs.append({
            "id": f"row_{idx}",
            "text": row_to_text(row),
            "metadata": {
                "crop": row["label"],
                "N": float(row["N"]),
                "P": float(row["P"]),
                "K": float(row["K"]),
                "temperature": float(row["temperature"]),
                "humidity": float(row["humidity"]),
                "ph": float(row["ph"]),
                "rainfall": float(row["rainfall"]),
                "chunk_type": "row",
                "source_row": int(idx),
            },
        })
    return docs


def build_aggregate_docs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Strategy 2: One document per crop summarising mean ± std and ranges."""
    docs = []
    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    for crop, grp in df.groupby("label"):
        stats = grp[numeric_cols].agg(["mean", "std", "min", "max"])
        text = (
            f"{crop.capitalize()} crop profile (n={len(grp)}):\n"
            f"  Nitrogen (N): mean={stats.loc['mean','N']:.1f}, "
            f"range=[{stats.loc['min','N']:.0f}–{stats.loc['max','N']:.0f}] kg/ha\n"
            f"  Phosphorus (P): mean={stats.loc['mean','P']:.1f}, "
            f"range=[{stats.loc['min','P']:.0f}–{stats.loc['max','P']:.0f}] kg/ha\n"
            f"  Potassium (K): mean={stats.loc['mean','K']:.1f}, "
            f"range=[{stats.loc['min','K']:.0f}–{stats.loc['max','K']:.0f}] kg/ha\n"
            f"  Temperature: mean={stats.loc['mean','temperature']:.1f}°C, "
            f"range=[{stats.loc['min','temperature']:.1f}–{stats.loc['max','temperature']:.1f}]°C\n"
            f"  Humidity: mean={stats.loc['mean','humidity']:.1f}%, "
            f"range=[{stats.loc['min','humidity']:.1f}–{stats.loc['max','humidity']:.1f}]%\n"
            f"  pH: mean={stats.loc['mean','ph']:.2f}, "
            f"range=[{stats.loc['min','ph']:.2f}–{stats.loc['max','ph']:.2f}]\n"
            f"  Rainfall: mean={stats.loc['mean','rainfall']:.1f}mm, "
            f"range=[{stats.loc['min','rainfall']:.1f}–{stats.loc['max','rainfall']:.1f}]mm"
        )
        docs.append({
            "id": f"agg_{crop}",
            "text": text,
            "metadata": {
                "crop": crop,
                "N_mean": float(stats.loc["mean", "N"]),
                "P_mean": float(stats.loc["mean", "P"]),
                "K_mean": float(stats.loc["mean", "K"]),
                "temperature_mean": float(stats.loc["mean", "temperature"]),
                "humidity_mean": float(stats.loc["mean", "humidity"]),
                "ph_mean": float(stats.loc["mean", "ph"]),
                "rainfall_mean": float(stats.loc["mean", "rainfall"]),
                "chunk_type": "aggregate",
                "n_samples": int(len(grp)),
            },
        })
    return docs


def build_hybrid_docs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Strategy 3: Aggregate docs + sampled row docs (10 rows per crop)."""
    agg_docs = build_aggregate_docs(df)
    row_docs = []
    for _, grp in df.groupby("label"):
        sample = grp.sample(min(10, len(grp)), random_state=42)
        row_docs.extend(build_row_level_docs(sample))
    return agg_docs + row_docs


# ── Public loader ─────────────────────────────────────────────────────────────

def load_and_chunk(
    csv_path: str,
    strategy: Literal["row", "aggregate", "hybrid"] = "hybrid",
) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['label'].nunique()} crops.")
    if strategy == "row":
        docs = build_row_level_docs(df)
    elif strategy == "aggregate":
        docs = build_aggregate_docs(df)
    else:
        docs = build_hybrid_docs(df)
    print(f"Strategy='{strategy}' → {len(docs)} documents.")
    return docs


if __name__ == "__main__":
    docs = load_and_chunk("data/Crop_recommendation.csv", strategy="hybrid")
    print(docs[0])
    print(docs[-1])
