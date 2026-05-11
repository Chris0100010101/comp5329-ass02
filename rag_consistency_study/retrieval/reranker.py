"""Reranker via LM Studio /v1/rerank endpoint (cross-encoder).

BGE-reranker-v2-m3 is a cross-encoder: it scores (query, document) pairs
jointly rather than embedding them separately. LM Studio exposes this via
the /v1/rerank endpoint, which is the correct way to use it.

Falls back to bi-encoder cosine similarity if /v1/rerank is unavailable.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import math
from pathlib import Path
from typing import List

import requests
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_loader import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_RERANK_MODEL,
    RERANKER_TOP_K,
)

_RERANK_ENDPOINT = f"{LM_STUDIO_BASE_URL.rstrip('/')}/rerank"
_EMBED_ENDPOINT  = f"{LM_STUDIO_BASE_URL.rstrip('/')}/embeddings"


def _rerank_cross_encoder(query: str, texts: List[str]) -> List[float]:
    """Call /v1/rerank — proper cross-encoder scoring."""
    payload = {
        "model": LM_STUDIO_RERANK_MODEL,
        "query": query,
        "documents": texts,
    }
    response = requests.post(_RERANK_ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    # results is a list of {index, relevance_score}
    scores = [0.0] * len(texts)
    for item in data["results"]:
        scores[item["index"]] = item["relevance_score"]
    return scores


def _embed(texts: List[str]) -> List[List[float]]:
    """Fallback: call /v1/embeddings with the reranker model."""
    payload = {"model": LM_STUDIO_RERANK_MODEL, "input": texts}
    response = requests.post(_EMBED_ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


def _cosine(a: List[float], b: List[float]) -> float:
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_reranker():
    """No-op — reranker is stateless (pure API calls). Returns None."""
    print(f"[OK] Reranker 就绪 (LM Studio cross-encoder): {LM_STUDIO_RERANK_MODEL}")
    return None


def rerank(
    query: str,
    documents: List[Document],
    top_k: int = RERANKER_TOP_K,
    model=None,
) -> List[Document]:
    """Re-rank documents using BGE reranker as cross-encoder.

    Tries /v1/rerank first (proper cross-encoder); falls back to
    bi-encoder cosine similarity if the endpoint is unavailable.
    """
    if not documents:
        return []

    doc_texts = [doc.page_content for doc in documents]

    try:
        scores = _rerank_cross_encoder(query, doc_texts)
        mode = "cross-encoder"
    except Exception as e:
        print(f"  ↳ /v1/rerank 不可用 ({e})，回退至 bi-encoder")
        embeds    = _embed([query] + doc_texts)
        query_vec = embeds[0]
        scores    = [_cosine(query_vec, dv) for dv in embeds[1:]]
        mode = "bi-encoder"

    scored_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )
    top_docs = [doc for _, doc in scored_docs[:top_k]]
    print(f"  ↳ Rerank({mode}) 后保留 {len(top_docs)}/{len(documents)} 条文档 "
          f"(top score={scored_docs[0][0]:.4f})")
    return top_docs
