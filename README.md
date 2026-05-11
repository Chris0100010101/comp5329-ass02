# COMP5329 Assignment 2

Empirical study on RAG pipeline architectures for multi-step research reasoning.

---

## Repository Structure

```
ass2/
├── data/                          # Corpus and topic files
│   ├── corpus.csv                 # 500 arXiv abstracts (retrieval corpus)
│   ├── topics.csv                 # 50 research topics (topic_id, topic)
│   └── ...
├── rag_consistency_study/         # Main experiment codebase
│   ├── main.py                    # Entry point
│   ├── pipelines/                 # vanilla / rag / rag_single / rag_rerank_single
│   ├── retrieval/                 # FAISS store, embeddings, reranker
│   ├── analysis/                  # LLM-as-judge consistency + evidence scoring
│   ├── prompts/                   # Prompt templates as editable .txt files
│   ├── data_processing/           # Corpus builder, result aggregator, Excel report
│   ├── utils/                     # Config loader, prompt loader
│   └── README.md                  # Detailed experiment documentation
├── Research Topic Idea - Enhanced.xlsx
└── Week3_Foundation.ipynb.pdf
```

---

## Research Question

Does the retrieval strategy affect the consistency and factual grounding of multi-step LLM research reasoning?

---

## Pipelines Compared

| System | Strategy |
|---|---|
| `vanilla` | No retrieval — parametric knowledge only |
| `rag` | Per-step retrieval — query drifts across steps |
| `rag_single` | Single retrieval at start — fixed context shared across all steps |
| `rag_rerank_single` | Single retrieval + BGE cross-encoder reranking — highest-precision shared context |

Each pipeline runs under two retrieval conditions (`clean` / `noisy`). Vanilla runs under `none`.

---

## Quick Start

```bash
cd rag_consistency_study

# 1. Configure API keys and endpoints
cp .env.example .env   # then fill in DEEPSEEK_API_KEY and LM_STUDIO_BASE_URL

# 2. Build FAISS index (once)
python main.py --build-corpus

# 3. Run a quick test (first 5 topics)
python main.py --limit 5

# 4. Run the full experiment (~2-3 hours)
python main.py
```

Results are saved to `rag_consistency_study/results/YYYYMMDD_HHMMSS/` and an Excel report is auto-generated to `rag_consistency_study/statistics/`.

See [`rag_consistency_study/README.md`](rag_consistency_study/README.md) for full documentation.

---

## Dependencies

```bash
pip install langchain-core langchain-openai langchain-community faiss-cpu \
            pandas openpyxl python-dotenv requests pydantic
```

Requires:
- **DeepSeek API** — generation and LLM-as-judge evaluation
- **LM Studio** (local) — BGE-M3 embeddings and BGE reranker

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Consistency (1-10) | Step-to-step coherence, internal consistency, completeness |
| Evidence (1-10) | Fraction of claims grounded in retrieved documents |
| Final score | `0.5 x Consistency + 0.5 x Evidence` |
| Coverage rate | Fraction of gaps with a matching hypothesis and experiment |
