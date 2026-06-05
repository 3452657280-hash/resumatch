"""应用配置 —— 统一管理 LLM、Embedding 和环境变量。"""
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class Settings(BaseSettings):
    dashscope_api_key: str = ""
    llm_model: str = "qwen-max"
    embedding_model: str = "text-embedding-v3"
    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma")
    upload_dir: str = str(BASE_DIR / "data" / "uploads")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()


class DashScopeLLM:
    """纯 httpx 实现的通义千问 LLM 客户端。"""

    def __init__(self, model: str, api_key: str, temperature: float = 0.1):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def invoke(self, messages: list[dict]) -> str:
        """messages: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        resp = httpx.post(
            f"{DASHSCOPE_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class DashScopeEmbedder:
    """纯 httpx 实现的通义千问 Embedding 客户端。"""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def embed(self, text: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": text}
        resp = httpx.post(
            f"{DASHSCOPE_BASE}/embeddings",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def get_llm(temperature: float = 0.1) -> DashScopeLLM:
    return DashScopeLLM(
        model=settings.llm_model,
        api_key=settings.dashscope_api_key,
        temperature=temperature,
    )


def get_embedder() -> DashScopeEmbedder:
    return DashScopeEmbedder(
        model=settings.embedding_model,
        api_key=settings.dashscope_api_key,
    )
