"""Build a single unified FAISS index from corpus.csv.

All 500 documents are indexed together with topic_id in metadata.
Clean vs noisy retrieval is controlled at query time via metadata filter.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.embeddings import LMStudioEmbeddings
from retrieval.faiss_store import build_faiss_store
from utils.config_loader import CORPUS_CSV


def load_corpus_csv(csv_path: Path = CORPUS_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[ERROR] corpus CSV 不存在: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    print(f"[OK] 加载 corpus: {len(df)} 篇文档 from {csv_path.name}")
    return df


def df_to_documents(df: pd.DataFrame) -> List[Document]:
    docs = []
    for _, row in df.iterrows():
        content = f"{row.get('title', '')}\n\n{row.get('abstract', '')}".strip()
        docs.append(Document(
            page_content=content,
            metadata={
                "doc_id":   str(row.get("doc_id", "")),
                "topic_id": str(row.get("topic_id", "")),
                "domain":   str(row.get("domain", "")),
            },
        ))
    return docs


def build_unified_corpus(embeddings: LMStudioEmbeddings | None = None):
    """Build a single FAISS index containing all corpus documents."""
    if embeddings is None:
        embeddings = LMStudioEmbeddings()

    df = load_corpus_csv()
    if df.empty:
        return None

    docs = df_to_documents(df)
    print(f"[OK] 共 {len(docs)} 篇文档，{df['topic_id'].nunique()} 个话题")
    return build_faiss_store(docs, embeddings=embeddings)


if __name__ == "__main__":
    build_unified_corpus()
