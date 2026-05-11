"""RAG Single-Retrieval pipeline.

Retrieves documents ONCE using the topic, then shares the same context
across all three reasoning steps. This eliminates per-step context drift
and should produce higher consistency scores than per-step RAG.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_loader import (
    get_deepseek_key, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL, RETRIEVAL_TOP_K
)
from utils.prompt_templates import (
    GAP_IDENTIFICATION_TEMPLATE,
    HYPOTHESIS_TEMPLATE,
    EXPERIMENT_DESIGN_TEMPLATE,
)
from retrieval.faiss_store import load_faiss_store, similarity_search


def _docs_to_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
    )


def build_rag_single_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEEPSEEK_CHAT_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=get_deepseek_key(),
        temperature=temperature,
    )


def run_rag_single_pipeline(
    topic: str,
    condition: str = "clean",
    topic_id: Optional[str] = None,
    llm: ChatOpenAI | None = None,
    vectorstore: FAISS | None = None,
    top_k: int = RETRIEVAL_TOP_K,
) -> Dict:
    """Three-step reasoning with a single retrieval shared across all steps.

    Args:
        topic: Research topic string.
        condition: "clean" (filter by topic_id) or "noisy" (no filter).
        topic_id: Metadata filter for clean condition.
        llm: ChatOpenAI instance; creates one if None.
        vectorstore: Pre-loaded FAISS store; loads from disk if None.
        top_k: Number of documents to retrieve (once, at the start).
    """
    if llm is None:
        llm = build_rag_single_llm()

    if vectorstore is None:
        vectorstore = load_faiss_store()
        if vectorstore is None:
            raise RuntimeError("[ERROR] FAISS 索引不存在，请先运行 --build-corpus")

    filter_id = topic_id if condition == "clean" else None
    parser = StrOutputParser()

    # ── Single retrieval using the topic ──────────────────────────────────────
    print(f"\n[RAG-Single/{condition}] 单次检索 — topic: {topic[:60]}")
    shared_docs = similarity_search(vectorstore, topic, top_k=top_k, topic_id=filter_id)
    shared_context = _docs_to_context(shared_docs)
    print(f"[OK] 固定文档集: {len(shared_docs)} 篇，三步共享")

    # ── Step 1: Gap Identification ─────────────────────────────────────────────
    print(f"[RAG-Single/{condition}] Step 1: Gap Identification")
    gap_chain = PromptTemplate.from_template(GAP_IDENTIFICATION_TEMPLATE) | llm | parser
    gap_analysis = gap_chain.invoke({"topic": topic, "context": shared_context})
    print(f"[OK] Gap analysis 完成 ({len(gap_analysis)} chars)")

    # ── Step 2: Hypothesis Generation ─────────────────────────────────────────
    print(f"[RAG-Single/{condition}] Step 2: Hypothesis Generation")
    hyp_chain = PromptTemplate.from_template(HYPOTHESIS_TEMPLATE) | llm | parser
    hypothesis = hyp_chain.invoke({
        "topic": topic,
        "gap_analysis": gap_analysis,
        "context": shared_context,   # 同一批文档
    })
    print(f"[OK] Hypothesis 完成 ({len(hypothesis)} chars)")

    # ── Step 3: Experiment Design ──────────────────────────────────────────────
    print(f"[RAG-Single/{condition}] Step 3: Experiment Design")
    exp_chain = PromptTemplate.from_template(EXPERIMENT_DESIGN_TEMPLATE) | llm | parser
    experiment_design = exp_chain.invoke({
        "topic": topic,
        "gap_analysis": gap_analysis,
        "hypothesis": hypothesis,
        "context": shared_context,   # 同一批文档
    })
    print(f"[OK] Experiment design 完成 ({len(experiment_design)} chars)")

    return {
        "system": "rag_single",
        "condition": condition,
        "topic": topic,
        "gap_analysis": gap_analysis,
        "hypothesis": hypothesis,
        "experiment_design": experiment_design,
        "retrieved_docs": [doc.page_content for doc in shared_docs],
    }
