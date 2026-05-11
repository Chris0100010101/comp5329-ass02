import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import json
import re
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_loader import get_deepseek_key, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL
from utils.prompt_templates import (
    CONSISTENCY_JUDGE_TEMPLATE,
    EVIDENCE_CONSISTENCY_TEMPLATE,
)


class ConsistencyScore(BaseModel):
    step_coherence_score: float = Field(ge=1, le=10)
    internal_consistency_score: float = Field(ge=1, le=10)
    completeness_score: float = Field(ge=1, le=10)
    contradictions: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @property
    def overall_score(self) -> float:
        return round(
            (self.step_coherence_score
             + self.internal_consistency_score
             + self.completeness_score) / 3,
            2,
        )


class EvidenceScore(BaseModel):
    evidence_alignment_score: float = Field(ge=1, le=10)
    supported_claims: int = 0
    partially_supported_claims: int = 0
    unsupported_claims: int = 0
    reasoning: str = ""


def _extract_json(raw: str) -> str:
    """Strip markdown code fences and extract the JSON object."""
    # Remove ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback: find first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)
    return raw


def _build_judge_llm() -> ChatOpenAI:
    api_key = get_deepseek_key()
    return ChatOpenAI(
        model=DEEPSEEK_CHAT_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        temperature=0.0,
    )


def evaluate_consistency(
    topic: str,
    gap_analysis: str,
    hypothesis: str,
    experiment_design: str,
    llm: Optional[ChatOpenAI] = None,
) -> ConsistencyScore:
    """Use LLM-as-judge to score step-to-step reasoning consistency."""
    if llm is None:
        llm = _build_judge_llm()

    prompt = PromptTemplate.from_template(CONSISTENCY_JUDGE_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    print("  ↳ 评估推理一致性...")
    raw = chain.invoke({
        "topic": topic,
        "gap_analysis": gap_analysis,
        "hypothesis": hypothesis,
        "experiment_design": experiment_design,
    })

    try:
        data = json.loads(_extract_json(raw))
        return ConsistencyScore(**data)
    except Exception as e:
        print(f"[WARN] 一致性评分解析失败，使用默认值: {e}")
        return ConsistencyScore(
            step_coherence_score=1,
            internal_consistency_score=1,
            completeness_score=1,
            contradictions=["解析失败"],
            reasoning=raw[:200],
        )


def evaluate_evidence_consistency(
    retrieved_docs: list[str],
    reasoning_output: str,
    llm: Optional[ChatOpenAI] = None,
) -> EvidenceScore:
    """Score whether reasoning claims are grounded in retrieved documents."""
    if llm is None:
        llm = _build_judge_llm()

    docs_text = "\n\n".join(
        f"[Doc {i+1}] {doc}" for i, doc in enumerate(retrieved_docs)
    )

    prompt = PromptTemplate.from_template(EVIDENCE_CONSISTENCY_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    print("  ↳ 评估证据一致性...")
    raw = chain.invoke({
        "retrieved_docs": docs_text,
        "reasoning_output": reasoning_output,
    })

    try:
        data = json.loads(_extract_json(raw))
        return EvidenceScore(**data)
    except Exception as e:
        print(f"[WARN] 证据评分解析失败，使用默认值: {e}")
        return EvidenceScore(
            evidence_alignment_score=1,
            reasoning=raw[:200],
        )
