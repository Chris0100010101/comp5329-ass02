# RAG Reasoning Consistency Study

COMP5329 Assignment 2 — Empirical study comparing reasoning consistency and evidence grounding across four RAG pipeline architectures.

---

## Research Question

Does the retrieval strategy (none / per-step / single-retrieval / reranked) affect the consistency and factual grounding of multi-step LLM research reasoning?

---

## Experimental Design

### Pipelines

| System | Description |
|---|---|
| `vanilla` | No retrieval. LLM generates entirely from parametric knowledge. |
| `rag` | Per-step retrieval. Query changes each step (topic → gap → hypothesis), causing context drift. |
| `rag_single` | Single retrieval at the start. Same 5 documents shared across all 3 steps. |
| `rag_rerank_single` | Single retrieval + BGE reranker reordering. Same reranked documents shared across all 3 steps. |

### Conditions

| Condition | Description |
|---|---|
| `none` | Vanilla only. No retrieval corpus involved. |
| `clean` | Filtered retrieval — only documents tagged to the current topic. |
| `noisy` | Unfiltered retrieval — full 500-document corpus, may include off-topic documents. |

### Reasoning Pipeline (3 steps)

1. **Gap Identification** — identify 3 research gaps given the topic
2. **Hypothesis Generation** — generate one testable hypothesis per gap
3. **Experiment Design** — design one experiment per hypothesis

### Evaluation

- **Consistency** (1–10): step-to-step coherence, internal consistency, completeness — scored by LLM-as-judge (DeepSeek, temperature=0)
- **Evidence** (1–10): fraction of claims grounded in retrieved documents — scored by LLM-as-judge
- **Final score**: `0.5 × consistency_overall + 0.5 × evidence`
- **Coverage rate**: structural check — fraction of gaps that have a matching hypothesis and experiment

---

## Directory Structure

```
rag_consistency_study/
├── main.py                         # entry point
├── .env                            # API keys and hyperparameters
├── faiss_index/
│   └── unified/                    # FAISS vector store (pre-built)
├── results/
│   └── YYYYMMDD_HHMMSS/           # one subfolder per run
│       ├── system_condition_TXX.json
│       └── summary.json
├── statistics/
│   └── YYYYMMDD_HHMMSS.xlsx       # auto-generated Excel report per run
├── pipelines/
│   ├── vanilla_pipeline.py
│   ├── rag_pipeline.py
│   ├── rag_single_pipeline.py
│   ├── rag_rerank_pipeline.py
│   └── rag_rerank_single_pipeline.py
├── retrieval/
│   ├── faiss_store.py              # FAISS load / similarity search
│   ├── embeddings.py               # LM Studio BGE-M3 embedding client
│   └── reranker.py                 # BGE reranker (cross-encoder via LM Studio)
├── analysis/
│   ├── consistency_evaluator.py    # LLM-as-judge scoring
│   └── contradiction_detector.py  # structural gap/hypothesis/experiment matching
├── data_processing/
│   ├── corpus_builder.py           # builds unified FAISS index from corpus.csv
│   ├── result_aggregator.py        # aggregates JSON results into summary table
│   └── statistics_generator.py    # generates Excel report with multiple sheets
└── utils/
    ├── config_loader.py            # loads .env, defines all paths and constants
    └── prompt_templates.py         # all LLM prompts (fixed across systems)
```

---

## Setup

### Requirements

```
python >= 3.11
langchain
langchain-openai
langchain-community
faiss-cpu
pandas
openpyxl
python-dotenv
requests
```

Install:

```bash
pip install langchain langchain-openai langchain-community faiss-cpu pandas openpyxl python-dotenv requests
```

### External Services

| Service | Purpose | Config |
|---|---|---|
| LM Studio (local) | BGE-M3 embedding + BGE reranker | `LM_STUDIO_BASE_URL` in `.env` |
| DeepSeek API | Generation + LLM-as-judge | `DEEPSEEK_API_KEY` in `.env` |

### .env

```
DEEPSEEK_API_KEY=<your key>

LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
EMBED_MODEL_NAME=text-embedding-bge-m3
LM_STUDIO_RERANK_MODEL=text-embedding-bge-reranker-v2-m3

RETRIEVAL_TOP_K=5
RERANKER_TOP_K=5
NOISE_RATIO=0.4
```

### Data

Data files live in `../data/` (one level above `rag_consistency_study/`):

```
ass2/
├── data/
│   ├── corpus.csv      # document corpus
│   └── topics.csv      # 50 research topics (columns: topic_id, topic)
└── rag_consistency_study/
```

---

## Usage

```bash
# Build FAISS index (required once before any experiment)
python main.py --build-corpus

# Run all systems x all conditions x all 50 topics (~2-3 hours)
python main.py

# Run a single system
python main.py --system rag_single

# Run a specific condition only
python main.py --condition clean

# Run first N topics (for testing)
python main.py --limit 5

# Run a single topic
python main.py --topic-id T001

# Aggregate and generate Excel from the latest run
python main.py --aggregate

# Aggregate a specific run folder
python main.py --aggregate --run-dir 20260509_121635
```

After each full run, results are saved to `results/YYYYMMDD_HHMMSS/` and an Excel report is automatically generated to `statistics/YYYYMMDD_HHMMSS.xlsx`.

---

## Excel Report Sheets

| Sheet | Content |
|---|---|
| Summary | Mean and std of all scores per system+condition, sorted by Evidence score |
| PerTopic | All individual records with color-scale on evidence and final score |
| CleanVsNoisy | Side-by-side clean vs noisy comparison with delta for each system |
| FinalHeatmap | Per-topic final score pivot table (color scale) |
| EvidenceHeatmap | Per-topic evidence score pivot table (color scale) |

---

## Key Findings (50 topics, full run)

| System | Evidence | Overall(consistency) | Final |
|---|---|---|---|
| rag_noisy | 3.06 | 5.114 | 4.087 |
| rag_single_clean | 2.92 | 5.174 | 4.047 |
| rag_rerank_single_noisy | 2.92 | 5.254 | 4.087 |
| rag_clean | 2.88 | 5.020 | 3.950 |
| rag_rerank_single_clean | 2.74 | 5.094 | 3.917 |
| rag_single_noisy | 2.78 | 5.253 | 4.017 |
| vanilla_none | 1.84 | 5.654 | 3.747 |

- `vanilla_none` scores highest on consistency (no retrieval pressure) but lowest on evidence (1.84), confirming hallucination without grounding.
- All RAG variants significantly improve evidence alignment over vanilla.
- Clean vs noisy difference is small (~0.1–0.3), suggesting robustness to retrieval noise within this corpus.
