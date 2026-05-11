"""Generate an Excel statistics report for a completed experiment run.

Usage (standalone):
    python statistics_generator.py                      # latest run
    python statistics_generator.py 20260509_103000      # specific run folder

Called automatically from main.py after each full run.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_DIR        = Path(__file__).resolve().parents[1]
RESULTS_BASE    = BASE_DIR / "results"
STATISTICS_DIR  = BASE_DIR / "statistics"

SCORE_FIELDS = [
    "step_coherence_score",
    "internal_consistency_score",
    "completeness_score",
    "overall_score",
    "evidence_alignment_score",
    "final_score",
    "n_contradictions",
    "coverage_rate",
]

SCORE_LABELS = {
    "step_coherence_score":       "Coherence",
    "internal_consistency_score": "Consistency",
    "completeness_score":         "Completeness",
    "overall_score":              "Overall(consistency)",
    "evidence_alignment_score":   "Evidence",
    "final_score":                "Final(0.5C+0.5E)",
    "n_contradictions":           "Contradictions",
    "coverage_rate":              "Coverage",
}

SYSTEM_ORDER = ["vanilla_none", "rag_clean", "rag_noisy",
                "rag_single_clean", "rag_single_noisy",
                "rag_rerank_single_clean", "rag_rerank_single_noisy"]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_records(run_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for f in sorted(run_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            records.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[WARN]  跳过 {f.name}: {e}")
    print(f"  加载 {len(records)} 条记录 from {run_dir.name}/")
    return records


def records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {
            "system":    r.get("system", "?"),
            "condition": r.get("condition", "?"),
            "topic_id":  r.get("topic_id", "?"),
            "topic":     r.get("topic", ""),
        }
        for f in SCORE_FIELDS:
            row[f] = r.get(f)
        # weighted composite: 50% consistency overall + 50% evidence
        ov = r.get("overall_score")
        ev = r.get("evidence_alignment_score")
        row["final_score"] = round(ov * 0.5 + ev * 0.5, 3) if (ov is not None and ev is not None) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ── sheet builders ─────────────────────────────────────────────────────────────

def build_summary_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each score field, grouped by system+condition. Sorted by Evidence desc."""
    df = df.copy()
    df["system_condition"] = df["system"] + "_" + df["condition"]

    score_fields_with_final = SCORE_FIELDS  # already includes final_score
    agg = (df.groupby("system_condition")[score_fields_with_final]
             .agg(["mean", "std"])
             .round(3))

    # flatten multi-level columns → "overall_score_mean" etc.
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    # add n (sample count)
    counts = df.groupby("system_condition")["overall_score"].count().rename("n")
    agg = agg.merge(counts, on="system_condition")

    # sort by evidence_alignment_score_mean descending (primary), final_score_mean (secondary)
    agg = agg.sort_values(
        ["evidence_alignment_score_mean", "final_score_mean"],
        ascending=False
    ).reset_index(drop=True)

    return agg


def build_per_topic_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """All individual records, sorted by system+condition then topic."""
    df = df.copy()
    df["system_condition"] = df["system"] + "_" + df["condition"]
    order = {k: i for i, k in enumerate(SYSTEM_ORDER)}
    df["_ord"] = df["system_condition"].map(lambda x: order.get(x, 99))
    df = df.sort_values(["_ord", "topic_id"]).drop(columns="_ord").reset_index(drop=True)
    return df


def build_clean_noisy_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side clean vs noisy delta for each system × score."""
    rows = []
    systems = ["rag", "rag_single", "rag_rerank_single"]
    for sys in systems:
        clean = df[(df["system"] == sys) & (df["condition"] == "clean")]
        noisy = df[(df["system"] == sys) & (df["condition"] == "noisy")]
        row = {"system": sys}
        for f in SCORE_FIELDS:
            c_mean = clean[f].mean()
            n_mean = noisy[f].mean()
            row[f"{SCORE_LABELS[f]}_clean"] = round(c_mean, 3) if pd.notna(c_mean) else None
            row[f"{SCORE_LABELS[f]}_noisy"] = round(n_mean, 3) if pd.notna(n_mean) else None
            if pd.notna(c_mean) and pd.notna(n_mean):
                row[f"{SCORE_LABELS[f]}_delta(C-N)"] = round(c_mean - n_mean, 3)
            else:
                row[f"{SCORE_LABELS[f]}_delta(C-N)"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def build_topic_heatmap_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Final score per topic × system_condition (pivot table, primary metric)."""
    df = df.copy()
    df["system_condition"] = df["system"] + "_" + df["condition"]
    pivot = df.pivot_table(
        index="topic_id", columns="system_condition",
        values="final_score", aggfunc="mean"
    ).round(3)
    # reorder columns
    cols = [c for c in SYSTEM_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    pivot = pivot.reset_index()
    return pivot


def build_evidence_heatmap_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Evidence score per topic × system_condition (pivot table)."""
    df = df.copy()
    df["system_condition"] = df["system"] + "_" + df["condition"]
    pivot = df.pivot_table(
        index="topic_id", columns="system_condition",
        values="evidence_alignment_score", aggfunc="mean"
    ).round(3)
    cols = [c for c in SYSTEM_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    pivot = pivot.reset_index()
    return pivot


# ── apply conditional formatting ──────────────────────────────────────────────

def _apply_color_scale(worksheet, df: pd.DataFrame, col_name: str,
                        col_idx: int, writer):
    """Green-white-red color scale on a numeric column (openpyxl)."""
    try:
        from openpyxl.formatting.rule import ColorScaleRule
        col_letter = chr(ord('A') + col_idx)
        n_rows = len(df)
        if n_rows < 2:
            return
        cell_range = f"{col_letter}2:{col_letter}{n_rows + 1}"
        rule = ColorScaleRule(
            start_type="min", start_color="F8696B",   # red
            mid_type="percentile", mid_value=50, mid_color="FFEB84",  # yellow
            end_type="max", end_color="63BE7B",        # green
        )
        worksheet.conditional_formatting.add(cell_range, rule)
    except Exception:
        pass  # formatting is optional


# ── main export ────────────────────────────────────────────────────────────────

def generate_excel(run_dir: Path) -> Path:
    STATISTICS_DIR.mkdir(exist_ok=True)
    out_path = STATISTICS_DIR / f"{run_dir.name}.xlsx"

    records = load_records(run_dir)
    if not records:
        print("[ERROR] 没有记录，跳过 Excel 生成")
        return out_path

    df = records_to_df(records)

    summary_df         = build_summary_sheet(df)
    per_topic_df       = build_per_topic_sheet(df)
    clean_noisy_df     = build_clean_noisy_sheet(df)
    heatmap_df         = build_topic_heatmap_sheet(df)
    evidence_heatmap_df = build_evidence_heatmap_sheet(df)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer,          sheet_name="Summary",         index=False)
        per_topic_df.to_excel(writer,        sheet_name="PerTopic",        index=False)
        clean_noisy_df.to_excel(writer,      sheet_name="CleanVsNoisy",    index=False)
        heatmap_df.to_excel(writer,          sheet_name="FinalHeatmap",    index=False)
        evidence_heatmap_df.to_excel(writer, sheet_name="EvidenceHeatmap", index=False)

        wb = writer.book

        # ── auto-width all sheets ──────────────────────────────────────────────
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for col in ws.columns:
                max_len = max(
                    (len(str(cell.value)) if cell.value is not None else 0)
                    for cell in col
                )
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

        # ── color scale on final_score + evidence in PerTopic ─────────────────
        ws_pt = wb["PerTopic"]
        cols_list = list(per_topic_df.columns)
        for target_col in ("final_score", "evidence_alignment_score"):
            if target_col in cols_list:
                _apply_color_scale(ws_pt, per_topic_df, target_col,
                                   cols_list.index(target_col), writer)

        # ── color scale on FinalHeatmap ────────────────────────────────────────
        ws_hm = wb["FinalHeatmap"]
        for i, col in enumerate(heatmap_df.columns):
            if col != "topic_id":
                _apply_color_scale(ws_hm, heatmap_df, col, i, writer)

        # ── color scale on EvidenceHeatmap ─────────────────────────────────────
        ws_eh = wb["EvidenceHeatmap"]
        for i, col in enumerate(evidence_heatmap_df.columns):
            if col != "topic_id":
                _apply_color_scale(ws_eh, evidence_heatmap_df, col, i, writer)

    print(f"[OK] Excel 已生成: {out_path}")
    return out_path


def run_statistics(run_dir: Path | None = None) -> Path | None:
    if run_dir is None:
        dirs = sorted([d for d in RESULTS_BASE.iterdir() if d.is_dir()])
        if not dirs:
            print("[ERROR] results/ 下没有任何运行目录")
            return None
        run_dir = dirs[-1]
        print(f"[OK] 自动选择最新运行目录: {run_dir.name}")
    return generate_excel(run_dir)


if __name__ == "__main__":
    import sys as _sys
    target = None
    if len(_sys.argv) > 1:
        target = RESULTS_BASE / _sys.argv[1]
        if not target.exists():
            print(f"[ERROR] 找不到目录: {target}")
            _sys.exit(1)
    run_statistics(target)
