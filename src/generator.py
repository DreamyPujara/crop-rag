"""
Generation Layer
Uses Anthropic Claude (claude-sonnet-4-6) to produce grounded answers
from retrieved crop-recommendation context.
"""

from __future__ import annotations
import os
from typing import List, Dict, Any

import anthropic


SYSTEM_PROMPT = """You are an expert agronomist assistant. Your role is to answer
questions about optimal crop recommendations using ONLY the provided dataset context.

Rules:
1. Base every claim strictly on the retrieved context passages below.
2. If the context is insufficient, say so explicitly rather than guessing.
3. Cite each data point you use by referencing its source (e.g. "[Row 42]" or "[Rice aggregate]").
4. Do not introduce external agronomic knowledge not present in the context.
5. When comparing crops, use the actual numerical values from context.
6. If the user asks about a condition not covered in context, acknowledge the gap.
"""


def build_context_block(hits: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = ["## Retrieved Context\n"]
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata", {})
        chunk_type = meta.get("chunk_type", "unknown")
        crop = meta.get("crop", "?")
        score = hit.get("score", 0.0)
        tag = f"[{i}] ({chunk_type}, crop={crop}, score={score:.3f})"
        lines.append(f"{tag}\n{hit['text']}\n")
    return "\n".join(lines)


def generate_answer(
    query: str,
    hits: List[Dict[str, Any]],
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
) -> Dict[str, Any]:
    """
    Call Claude with the retrieved context to generate a grounded answer.
    Returns dict with 'answer', 'context_used', 'citations'.
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    context_block = build_context_block(hits)
    user_message = (
        f"{context_block}\n\n"
        f"## Question\n{query}\n\n"
        f"## Instructions\nAnswer using ONLY the context above. "
        f"Cite the numbered passages you rely on (e.g. [1], [2])."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.content[0].text if response.content else ""

    # Extract citation numbers mentioned in the answer
    import re
    cited = list(set(re.findall(r"\[(\d+)\]", answer_text)))
    cited_docs = [hits[int(c) - 1] for c in cited if int(c) <= len(hits)]

    return {
        "query": query,
        "answer": answer_text,
        "context_used": hits,
        "citations": cited_docs,
        "model": model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def generate_answer_stream(
    query: str,
    hits: List[Dict[str, Any]],
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
):
    """Streaming version — yields text chunks."""
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    context_block = build_context_block(hits)
    user_message = (
        f"{context_block}\n\n## Question\n{query}\n\n"
        f"## Instructions\nAnswer using ONLY the context above. Cite passages used (e.g. [1])."
    )
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            yield text
