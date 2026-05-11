"""Cross-encoder reranker using sentence-transformers CrossEncoder.

BGE-reranker-v2-m3 is a cross-encoder: it scores (query, document) pairs
jointly for higher precision than bi-encoder similarity search.
Model is loaded from HuggingFace cache on first use and cached in memory.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import List

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_loader import LM_STUDIO_RERANK_MODEL, RERANKER_TOP_K

_CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_cross_encoder = None


def _get_cross_encoder():
    """Load CrossEncoder once and cache in memory."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print(f"  ↳ Loading cross-encoder: {_CROSS_ENCODER_MODEL_NAME}")
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL_NAME, max_length=512)
        print(f"  ↳ Cross-encoder loaded")
    return _cross_encoder


def load_reranker():
    """Pre-load the cross-encoder model at startup."""
    _get_cross_encoder()
    print(f"[OK] Reranker ready (cross-encoder): {_CROSS_ENCODER_MODEL_NAME}")
    return None


def rerank(
    query: str,
    documents: List[Document],
    top_k: int = RERANKER_TOP_K,
    model=None,
) -> List[Document]:
    """Re-rank documents using BGE cross-encoder.

    Args:
        query: The query string.
        documents: Candidate documents from FAISS retrieval.
        top_k: Number of top documents to return after reranking.
        model: Unused (kept for API compatibility).

    Returns:
        Top-k documents sorted by cross-encoder relevance score.
    """
    if not documents:
        return []

    cross_encoder = _get_cross_encoder()
    doc_texts = [doc.page_content for doc in documents]

    pairs = [(query, text) for text in doc_texts]
    scores = cross_encoder.predict(pairs)

    scored_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )
    top_docs = [doc for _, doc in scored_docs[:top_k]]
    top_score = float(scored_docs[0][0])
    print(f"  ↳ Rerank(cross-encoder) kept {len(top_docs)}/{len(documents)} docs "
          f"(top score={top_score:.4f})")
    return top_docs
