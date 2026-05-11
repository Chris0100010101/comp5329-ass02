"""RAG + Reranking Single-Retrieval pipeline.

Key design decisions vs rag_single:
1. Large candidate pool (RERANK_FETCH_K=20) fetched before reranking,
   so the cross-encoder actually has something meaningful to filter.
   rag_single fetches 5 and keeps all 5; this fetches 20 and keeps top 5.
2. Multi-query fusion: retrieves candidates with three complementary queries
   (topic / gap-focused expansion / hypothesis-focused expansion), deduplicates,
   then reranks the union in one pass. Wider coverage + higher precision.
3. Final reranked context is shared across all three reasoning steps (no drift).
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
    get_deepseek_key, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL,
    RETRIEVAL_TOP_K, RERANKER_TOP_K, RERANK_FETCH_K,
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


def _deduplicate(docs: List[Document]) -> List[Document]:
    seen, unique = set(), []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique


def _multi_query_candidates(
    vectorstore: FAISS,
    topic: str,
    fetch_k: int,
    filter_id: Optional[str],
) -> List[Document]:
    """Fetch candidates using three complementary query angles, then deduplicate.

    Query 1 (topic): broad topical coverage.
    Query 2 (research gap angle): biased toward gap/limitation language.
    Query 3 (method angle): biased toward experimental/methodology language.
    """
    q1 = topic
    q2 = f"research gaps limitations open problems {topic}"
    q3 = f"experimental methods evaluation benchmark {topic}"

    per_query_k = max(fetch_k // 2, RETRIEVAL_TOP_K)

    docs = []
    for q in [q1, q2, q3]:
        docs.extend(similarity_search(vectorstore, q, top_k=per_query_k, topic_id=filter_id))

    unique = _deduplicate(docs)
    print(f"  ↳ Multi-query fusion: 3 queries x {per_query_k} = {len(docs)} raw, "
          f"{len(unique)} unique candidates for reranking")
    return unique


def build_rag_rerank_single_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEEPSEEK_CHAT_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=get_deepseek_key(),
        temperature=temperature,
    )


def run_rag_rerank_single_pipeline(
    topic: str,
    condition: str = "clean",
    topic_id: Optional[str] = None,
    llm: ChatOpenAI | None = None,
    vectorstore: FAISS | None = None,
    reranker_model=None,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANKER_TOP_K,
    fetch_k: int = RERANK_FETCH_K,
) -> Dict:
    """Three-step reasoning with multi-query fusion + reranking, shared context.

    Improvement over rag_single:
    - Fetches fetch_k=20 candidates (vs 5) before reranking, giving the
      cross-encoder a meaningful selection task.
    - Multi-query fusion (3 angles) broadens recall before precision filtering.
    - Same reranked top-K shared across all three steps (no drift).

    Args:
        condition: "clean" (filter by topic_id) or "noisy" (no filter).
        topic_id: Metadata filter for clean condition.
        fetch_k: Candidates retrieved before reranking (should be >> rerank_top_k).
        rerank_top_k: Final documents kept after reranking.
    """
    if llm is None:
        llm = build_rag_rerank_single_llm()

    if vectorstore is None:
        vectorstore = load_faiss_store()
        if vectorstore is None:
            raise RuntimeError("[ERROR] FAISS index not found, run --build-corpus first")

    if reranker_model is None:
        reranker_model = load_reranker()

    filter_id = topic_id if condition == "clean" else None
    parser = StrOutputParser()

    # ── Multi-query retrieval + rerank ────────────────────────────────────────
    print(f"\n[RAG-Rerank-Single/{condition}] Multi-query retrieval + Rerank — topic: {topic[:60]}")
    candidates = _multi_query_candidates(vectorstore, topic, fetch_k, filter_id)

    # Rerank the full candidate pool against the topic query
    shared_docs = rerank(topic, candidates, top_k=rerank_top_k, model=reranker_model)
    shared_context = _docs_to_context(shared_docs)
    print(f"[OK] Fixed doc set: {len(shared_docs)} docs (reranked from {len(candidates)} candidates), shared across 3 steps")

    # ── Step 1: Gap Identification ─────────────────────────────────────────────
    print(f"[RAG-Rerank-Single/{condition}] Step 1: Gap Identification")
    gap_chain = PromptTemplate.from_template(GAP_IDENTIFICATION_TEMPLATE) | llm | parser
    gap_analysis = gap_chain.invoke({"topic": topic, "context": shared_context})
    print(f"[OK] Gap analysis done ({len(gap_analysis)} chars)")

    # ── Step 2: Hypothesis Generation ─────────────────────────────────────────
    print(f"[RAG-Rerank-Single/{condition}] Step 2: Hypothesis Generation")
    hyp_chain = PromptTemplate.from_template(HYPOTHESIS_TEMPLATE) | llm | parser
    hypothesis = hyp_chain.invoke({
        "topic": topic,
        "gap_analysis": gap_analysis,
        "context": shared_context,
    })
    print(f"[OK] Hypothesis done ({len(hypothesis)} chars)")

    # ── Step 3: Experiment Design ──────────────────────────────────────────────
    print(f"[RAG-Rerank-Single/{condition}] Step 3: Experiment Design")
    exp_chain = PromptTemplate.from_template(EXPERIMENT_DESIGN_TEMPLATE) | llm | parser
    experiment_design = exp_chain.invoke({
        "topic": topic,
        "gap_analysis": gap_analysis,
        "hypothesis": hypothesis,
        "context": shared_context,
    })
    print(f"[OK] Experiment design done ({len(experiment_design)} chars)")

    return {
        "system": "rag_rerank_single",
        "condition": condition,
        "topic": topic,
        "gap_analysis": gap_analysis,
        "hypothesis": hypothesis,
        "experiment_design": experiment_design,
        "retrieved_docs": [doc.page_content for doc in shared_docs],
    }
