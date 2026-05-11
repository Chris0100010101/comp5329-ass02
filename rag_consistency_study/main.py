"""Entry point for the RAG Reasoning Consistency Study.

Usage:
    python main.py                          # run all systems × all conditions
    python main.py --system vanilla         # single system
    python main.py --system rag_single      # single-retrieval RAG only
    python main.py --condition noisy        # single condition
    python main.py --topic-id T001          # single topic
    python main.py --build-corpus           # (re)build FAISS indexes only
    python main.py --aggregate              # aggregate saved results only
"""

import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import TOPICS_CSV, get_deepseek_key, RETRIEVAL_TOP_K
from retrieval.faiss_store import load_faiss_store, similarity_search
from retrieval.reranker import load_reranker
from retrieval.embeddings import LMStudioEmbeddings
from data_processing.corpus_builder import build_unified_corpus
from data_processing.result_aggregator import run_aggregation
from data_processing.statistics_generator import run_statistics
from pipelines.vanilla_pipeline import run_vanilla_pipeline, build_vanilla_llm
from pipelines.rag_pipeline import run_rag_pipeline
from pipelines.rag_single_pipeline import run_rag_single_pipeline
from pipelines.rag_rerank_pipeline import run_rag_rerank_pipeline
from pipelines.rag_rerank_single_pipeline import run_rag_rerank_single_pipeline
from analysis.consistency_evaluator import evaluate_consistency, evaluate_evidence_consistency
from analysis.contradiction_detector import summarize_cross_step_issues, count_contradictions

RESULTS_BASE = Path(__file__).resolve().parent / "results"
RESULTS_BASE.mkdir(exist_ok=True)

# 核心实验系统：vanilla(none) / rag / rag_single / rag_rerank_single
# 去掉 rag_rerank(逐步rerank) 和 vanilla_clean/noisy（与rag_single重叠）
SYSTEMS              = ["vanilla", "rag", "rag_single", "rag_rerank_single"]
CONDITIONS           = ["clean", "noisy"]
VANILLA_CONDITIONS   = ["none"]


def make_run_dir() -> Path:
    """Create a timestamped subfolder for this run, e.g. results/20260508_203000/"""
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 本次运行结果目录: {run_dir}")
    return run_dir


def load_topics() -> list[dict]:
    if not TOPICS_CSV.exists():
        print(f"[ERROR] topics.csv 不存在: {TOPICS_CSV}")
        return []
    df = pd.read_csv(TOPICS_CSV)
    topics = df[["topic_id", "topic"]].drop_duplicates().to_dict("records")
    print(f"[OK] 加载 {len(topics)} 个研究话题 from {TOPICS_CSV.name}")
    return topics


def save_result(result: dict, run_dir: Path):
    topic_id = result.get("topic_id", "T???")
    filename = run_dir / f"{result['system']}_{result['condition']}_{topic_id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OK] 结果已保存: {filename.name}")


def run_single(
    topic_id: str,
    topic: str,
    system: str,
    condition: str,
    run_dir: Path,
    llm=None,
    vectorstore=None,
    reranker_model=None,
    judge_llm=None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"System={system}  Condition={condition}  Topic={topic_id}")
    print(f"Topic: {topic}")
    print("=" * 60)

    if system == "vanilla":
        result = run_vanilla_pipeline(
            topic, condition=condition, topic_id=topic_id,
            llm=llm, vectorstore=vectorstore,
        )
    elif system == "rag":
        result = run_rag_pipeline(
            topic, condition=condition, topic_id=topic_id,
            llm=llm, vectorstore=vectorstore,
        )
    elif system == "rag_single":
        result = run_rag_single_pipeline(
            topic, condition=condition, topic_id=topic_id,
            llm=llm, vectorstore=vectorstore,
        )
    elif system == "rag_rerank":
        result = run_rag_rerank_pipeline(
            topic, condition=condition, topic_id=topic_id,
            llm=llm, vectorstore=vectorstore, reranker_model=reranker_model,
        )
    elif system == "rag_rerank_single":
        result = run_rag_rerank_single_pipeline(
            topic, condition=condition, topic_id=topic_id,
            llm=llm, vectorstore=vectorstore, reranker_model=reranker_model,
        )
    else:
        raise ValueError(f"Unknown system: {system}")

    result["topic_id"] = topic_id

    # ── Evaluation (consistency + evidence in parallel) ───────────────────────
    print("\n--- Evaluation ---")
    structural = summarize_cross_step_issues(
        result["gap_analysis"], result["hypothesis"], result["experiment_design"]
    )

    def _run_consistency():
        return evaluate_consistency(
            topic=topic,
            gap_analysis=result["gap_analysis"],
            hypothesis=result["hypothesis"],
            experiment_design=result["experiment_design"],
            llm=judge_llm,
        )

    def _run_evidence():
        # vanilla_none 没有检索文档，补一次 clean 检索专门用于评估
        eval_docs = result["retrieved_docs"]
        if not eval_docs and vectorstore is not None:
            eval_docs = [
                doc.page_content for doc in
                similarity_search(vectorstore, topic, top_k=RETRIEVAL_TOP_K, topic_id=topic_id)
            ]
            print("  ↳ vanilla 补充检索用于 evidence 评估")
        if not eval_docs:
            return None
        all_text = "\n\n".join([
            result["gap_analysis"],
            result["hypothesis"],
            result["experiment_design"],
        ])
        return evaluate_evidence_consistency(
            retrieved_docs=eval_docs,
            reasoning_output=all_text,
            llm=judge_llm,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_consistency = pool.submit(_run_consistency)
        f_evidence    = pool.submit(_run_evidence)
        consistency   = f_consistency.result()
        evidence_score = f_evidence.result()

    result.update({
        "step_coherence_score":        consistency.step_coherence_score,
        "internal_consistency_score":  consistency.internal_consistency_score,
        "completeness_score":          consistency.completeness_score,
        "overall_score":               consistency.overall_score,
        "contradictions":              consistency.contradictions,
        "n_contradictions":            count_contradictions(consistency.contradictions),
        "consistency_reasoning":       consistency.reasoning,
        "coverage_rate":               structural["coverage_rate"],
        "structural_analysis":         structural,
        "evidence_alignment_score":    evidence_score.evidence_alignment_score if evidence_score else None,
        "evidence_reasoning":          evidence_score.reasoning if evidence_score else None,
    })

    print(f"\n[OK] Overall={result['overall_score']}  "
          f"Evidence={result.get('evidence_alignment_score', 'N/A')}  "
          f"Coverage={result['coverage_rate']}")

    save_result(result, run_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description="RAG Reasoning Consistency Study")
    parser.add_argument("--system",     choices=SYSTEMS, default=None)
    parser.add_argument("--condition",  choices=CONDITIONS, default=None)
    parser.add_argument("--topic-id",   type=str, default=None,
                        help="Run a single topic by topic_id, e.g. T001")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Only run first N topics, e.g. --limit 10")
    parser.add_argument("--build-corpus", action="store_true")
    parser.add_argument("--aggregate",    action="store_true")
    parser.add_argument("--run-dir",      type=str, default=None,
                        help="Specify a run subfolder to aggregate, e.g. 20260508_210000")
    args = parser.parse_args()

    if args.aggregate:
        if args.run_dir:
            target = RESULTS_BASE / args.run_dir
        else:
            # pick latest timestamped subfolder; fall back to flat results/ itself
            run_dirs = sorted([d for d in RESULTS_BASE.iterdir() if d.is_dir()])
            target = run_dirs[-1] if run_dirs else RESULTS_BASE
        print(f"[OK] 聚合目录: {target}")
        run_aggregation(results_dir=target)
        run_statistics(run_dir=target)
        return

    all_topics = load_topics()
    if not all_topics:
        return

    topics = (
        [t for t in all_topics if t["topic_id"] == args.topic_id]
        if args.topic_id else all_topics
    )
    if not topics:
        print(f"[ERROR] 找不到 topic_id={args.topic_id}")
        return

    if args.limit:
        topics = topics[:args.limit]
        print(f"[OK] --limit {args.limit}: 只跑前 {len(topics)} 个话题")

    if args.build_corpus:
        print("\n[...] 构建统一 FAISS 索引（所有话题）...")
        build_unified_corpus()
        return

    systems    = [args.system]    if args.system    else SYSTEMS
    conditions = [args.condition] if args.condition else CONDITIONS

    print("\n[...] 加载共享资源...")
    llm       = build_vanilla_llm()
    judge_llm = build_vanilla_llm(temperature=0.0)
    embeddings = LMStudioEmbeddings()

    vectorstore = load_faiss_store(embeddings=embeddings)
    if vectorstore is None:
        print("[ERROR] 请先运行 --build-corpus")
        return

    reranker_model = None
    if any("rag_rerank" in s for s in systems):
        reranker_model = load_reranker()

    run_dir = make_run_dir()

    all_results = []
    for t in topics:
        topic_id = t["topic_id"]
        topic    = t["topic"]

        for system in systems:
            applicable_conditions = VANILLA_CONDITIONS if system == "vanilla" else conditions
            for condition in applicable_conditions:
                result = run_single(
                    topic_id=topic_id,
                    topic=topic,
                    system=system,
                    condition=condition,
                    run_dir=run_dir,
                    llm=llm,
                    vectorstore=vectorstore,
                    reranker_model=reranker_model,
                    judge_llm=judge_llm,
                )
                all_results.append(result)

    print(f"\n[OK] 全部实验完成，共 {len(all_results)} 条记录")
    run_aggregation(results_dir=run_dir)
    run_statistics(run_dir=run_dir)


if __name__ == "__main__":
    main()
