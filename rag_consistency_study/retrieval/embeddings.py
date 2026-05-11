import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import List

import requests
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_loader import LM_STUDIO_BASE_URL, EMBED_MODEL_NAME


class LMStudioEmbeddings(Embeddings):
    """Calls LM Studio's local OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self,
        base_url: str = LM_STUDIO_BASE_URL,
        model_name: str = EMBED_MODEL_NAME,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._endpoint = f"{self.base_url}/embeddings"

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model_name, "input": texts}
        try:
            response = requests.post(self._endpoint, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] 无法连接 LM Studio，请确认服务已启动: {self._endpoint}")
            raise
        except Exception as e:
            print(f"[ERROR] Embedding API 调用失败: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f" 嵌入 {len(texts)} 条文档 (model={self.model_name})")
        return self._call_api(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0]
