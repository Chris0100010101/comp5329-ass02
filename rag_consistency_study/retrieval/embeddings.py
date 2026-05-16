"""Local BGE-M3 embeddings via sentence-transformers (no LM Studio required).

Model: BAAI/bge-m3
Loaded once at first use and cached in memory.
Drop-in replacement for the previous LMStudioEmbeddings class —
implements the same LangChain Embeddings interface.

Note: if you previously built a FAISS index with LM Studio's GGUF version
of BGE-M3, rebuild it with --build-corpus to ensure vector consistency.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_EMBED_MODEL_NAME = "BAAI/bge-m3"
_model = None


def _get_model():
    """Load SentenceTransformer once and cache in memory."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  [Embeddings] Loading {_EMBED_MODEL_NAME} ...")
        _model = SentenceTransformer(_EMBED_MODEL_NAME)
        print(f"  [Embeddings] Model ready (dim={_model.get_sentence_embedding_dimension()})")
    return _model


class LocalBGEEmbeddings(Embeddings):
    """Local BGE-M3 embeddings using sentence-transformers CrossEncoder."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = _get_model()
        print(f"  [Embeddings] Encoding {len(texts)} documents ...")
        vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = _get_model()
        vec = model.encode([text], show_progress_bar=False, normalize_embeddings=True)
        return vec[0].tolist()


# Keep old name as alias so existing imports still work
LMStudioEmbeddings = LocalBGEEmbeddings
