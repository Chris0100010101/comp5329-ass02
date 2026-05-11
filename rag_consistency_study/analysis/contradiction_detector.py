import re
from typing import List


def extract_numbered_items(text: str) -> List[str]:
    """Split a numbered list response into individual items."""
    pattern = re.compile(r"(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\Z)", re.DOTALL)
    matches = pattern.findall(text)
    return [m.strip() for m in matches] if matches else [text.strip()]


def detect_coverage_gaps(
    gaps: List[str],
    hypotheses: List[str],
    experiments: List[str],
) -> dict:
    """Check whether hypotheses and experiments cover all identified gaps.

    Returns a dict with:
        - uncovered_gaps: gap indices with no matching hypothesis
        - untested_hypotheses: hypothesis indices with no matching experiment
        - coverage_rate: fraction of gaps addressed end-to-end
    """
    n_gaps = len(gaps)
    n_hyp  = len(hypotheses)
    n_exp  = len(experiments)

    uncovered_gaps       = list(range(n_hyp, n_gaps))
    untested_hypotheses  = list(range(n_exp, n_hyp))

    covered = n_gaps - len(uncovered_gaps)
    coverage_rate = round(covered / n_gaps, 2) if n_gaps else 0.0

    return {
        "n_gaps": n_gaps,
        "n_hypotheses": n_hyp,
        "n_experiments": n_exp,
        "uncovered_gaps": uncovered_gaps,
        "untested_hypotheses": untested_hypotheses,
        "coverage_rate": coverage_rate,
    }


def count_contradictions(contradictions: List[str]) -> int:
    """Return the number of non-trivial contradictions in a judge's list."""
    return sum(1 for c in contradictions if c.strip().lower() not in ("none", ""))


def summarize_cross_step_issues(
    gap_analysis: str,
    hypothesis: str,
    experiment_design: str,
) -> dict:
    """Lightweight structural analysis of cross-step coverage."""
    gaps        = extract_numbered_items(gap_analysis)
    hypotheses  = extract_numbered_items(hypothesis)
    experiments = extract_numbered_items(experiment_design)

    coverage = detect_coverage_gaps(gaps, hypotheses, experiments)

    print(f"  ↳ 结构分析: {coverage['n_gaps']} gaps → "
          f"{coverage['n_hypotheses']} hypotheses → "
          f"{coverage['n_experiments']} experiments "
          f"(coverage={coverage['coverage_rate']})")

    return coverage
