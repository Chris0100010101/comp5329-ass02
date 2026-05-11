"""Vanilla LLM pipeline — three conditions:

  none  : pure LLM, no retrieval context (baseline)
  clean : LLM + clean retrieved docs injected (topic-filtered)
  noisy : LLM + noisy retrieved docs injected (no filter)

Comparing none vs clean vs noisy isolates the pure effect of context
quality on reasoning, independent of pipeline architecture.
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
    GAP_IDENTIFICATION_TEMPLATE_VANILLA,
    HYPOTHESIS_TEMPLATE_VANILLA,
    EXPERIMENT_DESIGN_TEMPLATE_VANILLA,
)
from retrieval.faiss_store import load_faiss_store, similarity_search


def _docs_to_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
    )


def build_vanilla_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEEPSEEK_CHAT_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=get_deepseek_key(),
        temperature=temperature,
    )


def run_vanilla_pipeline(
    topic: str,
    condition: str = "none",
    topic_id: Optional[str] = None,
    llm: ChatOpenAI | None = None,
    vectorstore: FAISS | None = None,
    top_k: int = RETRIEVAL_TOP_K,
) -> Dict[str, str]:
    """Run the three-step reasoning pipeline.

    Args:
        condition: "none" (no context), "clean" (topic-filtered docs),
                   "noisy" (unfiltered docs from full corpus).
        topic_id: Required for clean condition.
    """
    if llm is None:
        llm = build_vanilla_llm()

    parser = StrOutputParser()
    retrieved_docs: List[Document] = []

    # ── Retrieve context once if condition != none ─────────────────────────────
    if condition in ("clean", "noisy"):
        if vectorstore is None:
            vectorstore = load_faiss_store()
            if vectorstore is None:
                raise RuntimeError("[ERROR] FAISS 索引不存在，请先运行 --build-corpus")
        filter_id = topic_id if condition == "clean" else None
        retrieved_docs = similarity_search(vectorstore, topic, top_k=top_k, topic_id=filter_id)
        shared_context = _docs_to_context(retrieved_docs)
        print(f"\n[Vanilla/{condition}] 注入 {len(retrieved_docs)} 篇文档作为上下文")
        gap_tmpl  = GAP_IDENTIFICATION_TEMPLATE
        hyp_tmpl  = HYPOTHESIS_TEMPLATE
        exp_tmpl  = EXPERIMENT_DESIGN_TEMPLATE
    else:
        shared_context = None
        print(f"\n[Vanilla/none] 无检索上下文")
        gap_tmpl  = GAP_IDENTIFICATION_TEMPLATE_VANILLA
        hyp_tmpl  = HYPOTHESIS_TEMPLATE_VANILLA
        exp_tmpl  = EXPERIMENT_DESIGN_TEMPLATE_VANILLA

    label = f"Vanilla/{condition}"

    # ── Step 1: Gap Identification ────────────────────────────────────────────
    print(f"[{label}] Step 1: Gap Identification — topic: {topic[:60]}")
    gap_chain = PromptTemplate.from_template(gap_tmpl) | llm | parser
    gap_inputs = {"topic": topic}
    if shared_context is not None:
        gap_inputs["context"] = shared_context
    gap_analysis = gap_chain.invoke(gap_inputs)
    print(f"[OK] Gap analysis 完成 ({len(gap_analysis)} chars)")

    # ── Step 2: Hypothesis Generation ─────────────────────────────────────────
    print(f"[{label}] Step 2: Hypothesis Generation")
    hyp_chain = PromptTemplate.from_template(hyp_tmpl) | llm | parser
    hyp_inputs = {"topic": topic, "gap_analysis": gap_analysis}
    if shared_context is not None:
        hyp_inputs["context"] = shared_context
    hypothesis = hyp_chain.invoke(hyp_inputs)
    print(f"[OK] Hypothesis 完成 ({len(hypothesis)} chars)")

    # ── Step 3: Experiment Design ──────────────────────────────────────────────
    print(f"[{label}] Step 3: Experiment Design")
    exp_chain = PromptTemplate.from_template(exp_tmpl) | llm | parser
    exp_inputs = {"topic": topic, "gap_analysis": gap_analysis, "hypothesis": hypothesis}
    if shared_context is not None:
        exp_inputs["context"] = shared_context
    experiment_design = exp_chain.invoke(exp_inputs)
    print(f"[OK] Experiment design 完成 ({len(experiment_design)} chars)")

    return {
        "system": "vanilla",
        "condition": condition,
        "topic": topic,
        "gap_analysis": gap_analysis,
        "hypothesis": hypothesis,
        "experiment_design": experiment_design,
        "retrieved_docs": [doc.page_content for doc in retrieved_docs],
    }
