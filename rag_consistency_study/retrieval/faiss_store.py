import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_loader import FAISS_DIR, RETRIEVAL_TOP_K
from retrieval.embeddings import LMStudioEmbeddings

UNIFIED_INDEX_DIR = FAISS_DIR / "unified"


def build_faiss_store(
    documents: List[Document],
    embeddings: LMStudioEmbeddings | None = None,
) -> FAISS:
    """Build and persist a single unified FAISS index for all documents."""
    if embeddings is None:
        embeddings = LMStudioEmbeddings()

    UNIFIED_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[OK] 构建统一 FAISS 索引，文档总数: {len(documents)}")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(UNIFIED_INDEX_DIR))
    print(f"[OK] FAISS 索引已保存至: {UNIFIED_INDEX_DIR}")
    return vectorstore


def load_faiss_store(
    embeddings: LMStudioEmbeddings | None = None,
) -> FAISS | None:
    """Load the unified FAISS index."""
    if embeddings is None:
        embeddings = LMStudioEmbeddings()

    if not (UNIFIED_INDEX_DIR / "index.faiss").exists():
        print(f"[ERROR] FAISS 索引不存在: {UNIFIED_INDEX_DIR}，请先运行 --build-corpus")
        return None

    vectorstore = FAISS.load_local(
        str(UNIFIED_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[OK] 已加载统一 FAISS 索引 from {UNIFIED_INDEX_DIR}")
    return vectorstore


def similarity_search(
    vectorstore: FAISS,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    topic_id: Optional[str] = None,
) -> List[Document]:
    """Retrieve documents with optional topic_id filter.

    Args:
        topic_id: If provided (clean condition), only return docs with matching
                  topic_id. If None (noisy condition), return from full corpus.
    """
    if topic_id:
        # Clean: filter to same topic, fallback to global if insufficient
        docs = vectorstore.similarity_search(
            query, k=top_k, filter={"topic_id": topic_id}
        )
        if len(docs) < top_k:
            print(f"  ↳ filter 结果不足 ({len(docs)})，补充全局召回")
            extra = vectorstore.similarity_search(query, k=top_k * 2)
            seen = {d.page_content for d in docs}
            for d in extra:
                if d.page_content not in seen:
                    docs.append(d)
                    seen.add(d.page_content)
                if len(docs) >= top_k:
                    break
        print(f"  ↳ 检索到 {len(docs)} 条文档 [clean, topic={topic_id}]")
    else:
        # Noisy: no filter, natural mix from full corpus
        docs = vectorstore.similarity_search(query, k=top_k)
        print(f"  ↳ 检索到 {len(docs)} 条文档 [noisy, no filter]")

    return docs
