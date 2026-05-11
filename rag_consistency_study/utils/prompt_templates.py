"""Load prompt templates from prompts/ directory.

Each prompt lives in its own .txt file so it can be edited without touching
any Python code. Templates are loaded once at import time and cached.

Placeholder syntax follows LangChain PromptTemplate.from_template():
  {variable}   -- substituted at runtime
  {{  }}       -- literal braces in the output (e.g. JSON schemas)
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _load(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Expected location: prompts/{filename}"
        )
    return path.read_text(encoding="utf-8")


# ── Step 1: Research Gap Identification ──────────────────────────────────────
GAP_IDENTIFICATION_TEMPLATE         = _load("gap_identification.txt")
GAP_IDENTIFICATION_TEMPLATE_VANILLA = _load("gap_identification_vanilla.txt")

# ── Step 2: Hypothesis Generation ────────────────────────────────────────────
HYPOTHESIS_TEMPLATE         = _load("hypothesis.txt")
HYPOTHESIS_TEMPLATE_VANILLA = _load("hypothesis_vanilla.txt")

# ── Step 3: Experiment Design ─────────────────────────────────────────────────
EXPERIMENT_DESIGN_TEMPLATE         = _load("experiment_design.txt")
EXPERIMENT_DESIGN_TEMPLATE_VANILLA = _load("experiment_design_vanilla.txt")

# ── Judges ────────────────────────────────────────────────────────────────────
CONSISTENCY_JUDGE_TEMPLATE   = _load("consistency_judge.txt")
EVIDENCE_CONSISTENCY_TEMPLATE = _load("evidence_judge.txt")
