"""Aggregate per-run evaluation results into a summary table."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_DIR      = Path(__file__).resolve().parents[1]
RESULTS_BASE  = BASE_DIR / "results"

SCORE_FIELDS = [
    "step_coherence_score",
    "internal_consistency_score",
    "completeness_score",
    "overall_score",
    "evidence_alignment_score",
    "n_contradictions",
    "coverage_rate",
]


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON result files from a run directory (no dedup needed — one file per topic)."""
    records = []
    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name == "summary.json":
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as e:
            print(f"[WARN] 读取结果文件失败 {json_file.name}: {e}")
    print(f"[OK] 加载 {len(records)} 条实验结果 from {results_dir.name}/")
    return records


def aggregate_by_system_condition(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Average scores grouped by (system, condition)."""
    groups: Dict[str, List[Dict]] = {}

    for rec in records:
        key = f"{rec.get('system', '?')}_{rec.get('condition', '?')}"
        groups.setdefault(key, []).append(rec)

    summary = {}
    for key, recs in groups.items():
        n = len(recs)
        avg = {}
        for field in SCORE_FIELDS:
            values = [r.get(field) for r in recs if r.get(field) is not None]
            avg[field] = round(sum(values) / len(values), 3) if values else None
        summary[key] = {"n_runs": n, **avg}

    return summary


def save_summary(summary: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] 摘要已保存至: {output_path}")


def print_summary_table(summary: Dict[str, Any]):
    """Print a simple ASCII table of aggregated scores."""
    header = (
        f"{'System+Condition':<25} {'Coherence':>10} {'Consistency':>12}"
        f" {'Completeness':>13} {'Overall':>9} {'Evidence':>9}"
        f" {'Contradictions':>15} {'Coverage':>9}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<25}"
            f" {str(row.get('step_coherence_score', '-')):>10}"
            f" {str(row.get('internal_consistency_score', '-')):>12}"
            f" {str(row.get('completeness_score', '-')):>13}"
            f" {str(row.get('overall_score', '-')):>9}"
            f" {str(row.get('evidence_alignment_score', '-')):>9}"
            f" {str(row.get('n_contradictions', '-')):>15}"
            f" {str(row.get('coverage_rate', '-')):>9}"
        )
    print("=" * len(header) + "\n")


def run_aggregation(results_dir: Path | None = None):
    """Aggregate results from results_dir. If None, uses the latest run subfolder."""
    if results_dir is None:
        run_dirs = sorted([d for d in RESULTS_BASE.iterdir() if d.is_dir()])
        if not run_dirs:
            print("[ERROR] results/ 下没有任何运行目录")
            return None
        results_dir = run_dirs[-1]
        print(f"[OK] 自动选择最新运行目录: {results_dir.name}")

    records = load_results(results_dir)
    if not records:
        print("[ERROR] 没有可聚合的结果，请先运行实验")
        return None

    summary = aggregate_by_system_condition(records)
    save_summary(summary, results_dir / "summary.json")
    print_summary_table(summary)
    return summary


if __name__ == "__main__":
    run_aggregation()
