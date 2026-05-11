import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)

# ── LM Studio (local embedding server) ──────────────────────────────────────
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
EMBED_MODEL_NAME   = os.environ.get("EMBED_MODEL_NAME", "text-embedding-bge-m3")

# ── DeepSeek ─────────────────────────────────────────────────────────────────
DEEPSEEK_BASE_URL       = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL     = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"

# ── Reranker —— LM Studio 本地服务（OpenAI-compat rerank endpoint） ───────────
LM_STUDIO_RERANK_MODEL = os.environ.get("LM_STUDIO_RERANK_MODEL", "text-embedding-bge-reranker-v2-m3")
RERANKER_TOP_K         = int(os.environ.get("RERANKER_TOP_K", "5"))
RERANK_FETCH_K         = int(os.environ.get("RERANK_FETCH_K", "20"))  # candidates fetched before reranking

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parents[1]          # rag_consistency_study/
DATA_DIR        = BASE_DIR.parent / "data"                     # ass2/data/
CORPUS_CSV      = DATA_DIR / "corpus.csv"
TOPICS_CSV      = DATA_DIR / "topics.csv"

# ── FAISS ─────────────────────────────────────────────────────────────────────
FAISS_DIR       = BASE_DIR / "faiss_index"
FAISS_CLEAN_DIR = FAISS_DIR / "clean"
FAISS_NOISY_DIR = FAISS_DIR / "noisy"

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "10"))

# ── Corpus building ───────────────────────────────────────────────────────────
NOISE_RATIO = float(os.environ.get("NOISE_RATIO", "0.4"))     # 移入此处统一管理


def get_deepseek_key() -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("[ERROR] 未找到 DEEPSEEK_API_KEY，请在 .env 文件中配置")
    return key
