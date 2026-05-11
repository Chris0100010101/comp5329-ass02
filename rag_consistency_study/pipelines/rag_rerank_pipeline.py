"""RAG + Reranking pipeline — unified FAISS with topic_id filter + BGE reranking."""

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
    get_deepseek_key, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL,
    RETRIEVAL_TOP_K, RERANKER_TOP_K,
)
from utils.prompt_templates import (
    GAP_IDENTIFICATION_TEMPLATE,
    HYPOTHESIS_TEMPLATE,
    EXPERIMENT_DESIGN_TEMPLATE,
)
from retrieval.faiss_store import load_faiss_store, similarity_search
from retrieval.reranker import rerank, load_reranker


def _docs_to_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
    )


def build_rerank_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEEPSEEK_CHAT_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=get_deepseek_key(),
        temperature=temperature,
    )


def run_rag_rerank_pipeline(
    topic: str,
    condition: str = "clean",
    topic_id: Optional[str] = None,
    llm: ChatOpenAI | None = None,
    vectorstore: FAISS | None = None,
    reranker_model=None,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANKER_TOP_K,
) -> Dict:
    """Run the three-step reasoning pipeline with unified retrieval + BGE reranking.

    Args:
        condition: "clean" (filter by topic_id) or "noisy" (no filter).
        topic_id: Used as metadata filter for clean condition.
    """
    if llm is None:
        llm = build_rerank_llm()

    if vectorstore is None:
        vectorstore = load_faiss_store()
        if vectorstore is None:
            raise RuntimeError("[ERROR] FAISS 索引不存在，请先运行 --build-corpus")

    if reranker_model is None:
        reranker_model = load_reranker()

    filter_id = topic_id if condition == "clean" else None

    parser = StrOutputParser()
    all_retrieved: List[Document] = []

    def retrieve_and_rerank(query: str) -> tuple[List[Document], str]:
        candidates = similarity_search(vectorstore, query, top_k=retrieval_top_k, topic_id=filter_id)
        reranked   = rerank(query, candidates, top_k=rerank_top_k, model=reranker_model)
        return reranked, _docs_to_context(reranked)

    # ── Step 1: Gap Identification ─────────────────────────────────────────────
    print(f"\n[RAG+Rerank/{condition}] Step 1: Gap Identification — topic: {topic[:60]}")
    docs1, context1 = retrieve_and_rerank(topic)
    all_retrieved.extend(docs1)

    gap_chain = PromptTemplate.from_template(GAP_IDENTIFICATION_TEMPLATE) | llm | parser
    gap_analysis = gap_chain.invoke({"topic": topic, "context": context1})
    print(f"[OK] Gap analysis 完成 ({len(gap_analysis)} chars)")

    # ── Step 2: Hypothesis Generation ─────────────────────────────────────────
    print(f"[RAG+Rerank/{condition}] Step 2: Hypothesis Generation")
    docs2, context2 = retrieve_and_rerank(gap_analysis)
    all_retrieved.extend(docs2)

    hyp_chain = PromptTemplate.from_template(HYPOTHESIS_TEMPLATE) | llm | parser
    hypothesis = hyp_chain.invoke({
        "topic": topic,
        "gap_analysis": gap_analysis,
        "context": context2,
    })
    print(f"[OK] Hypothesis 完成 ({len(hypothesis)} chars)")

    # ── Step 3: Experiment Design ──────────────────────────────────────────────
    print(f"[RAG+Rerank/{condition}] Step 3: Experiment Design")
    docs3, context3 = retrieve_and_rerank(hypothesis)
    all_retrieved.extend(docs3)

    exp_chain = PromptTemplate.from_template(EXPERIMENT_DESIGN_TEMPLATE) | llm | parser
    experiment_design = exp_chain.invoke({
        "topic": topic,
        "gap_analysis": gap_analysis,
        "hypothesis": hypothesis,
        "context": context3,
    })
    print(f"[OK] Experiment design 完成 ({len(experiment_design)} chars)")

    seen, unique_docs = set(), []
    for doc in all_retrieved:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    return {
        "system": "rag_rerank",
        "condition": condition,
        "topic": topic,
        "gap_analysis": gap_analysis,
        "hypothesis": hypothesis,
        "experiment_design": experiment_design,
        "retrieved_docs": [doc.page_content for doc in unique_docs],
    }
