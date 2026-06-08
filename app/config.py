"""应用配置 —— 统一管理 LLM、Embedding 和环境变量。"""
from pathlib import Path
from typing import Any

import json
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

    def invoke(self, messages: list[dict], max_retries: int = 2) -> str:
        """messages: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        
        带自动重试，失败抛出 RuntimeError（含友好错误信息）。
        """
        if not self.api_key:
            raise RuntimeError("通义千问 API Key 未配置，请在 .env 文件中设置 DASHSCOPE_API_KEY")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        last_error = None
        for attempt in range(1 + max_retries):
            try:
                resp = httpx.post(
                    f"{DASHSCOPE_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except httpx.TimeoutException:
                last_error = f"LLM 请求超时（已重试 {attempt}/{max_retries} 次）"
                if attempt < max_retries:
                    continue
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 401:
                    raise RuntimeError("通义千问 API Key 无效，请检查 .env 文件中的 DASHSCOPE_API_KEY")
                elif status == 429:
                    last_error = "LLM 请求过于频繁，请稍后重试"
                    if attempt < max_retries:
                        continue
                else:
                    raise RuntimeError(f"LLM API 返回异常 (HTTP {status})，请稍后重试")
            except httpx.ConnectError:
                last_error = "无法连接通义千问 API，请检查网络"
                if attempt < max_retries:
                    continue
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                raise RuntimeError(f"LLM 返回数据格式异常: {e}")

        raise RuntimeError(last_error or "LLM 调用失败，请稍后重试")


class DashScopeEmbedder:
    """纯 httpx 实现的通义千问 Embedding 客户端。"""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def embed(self, text: str, max_retries: int = 2) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": text}

        last_error = None
        for attempt in range(1 + max_retries):
            try:
                resp = httpx.post(
                    f"{DASHSCOPE_BASE}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
            except httpx.TimeoutException:
                last_error = f"Embedding 请求超时（已重试 {attempt}/{max_retries} 次）"
                if attempt < max_retries:
                    continue
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 401:
                    raise RuntimeError("通义千问 API Key 无效，Embedding 服务认证失败")
                elif status == 429:
                    last_error = "Embedding 请求过于频繁，请稍后重试"
                    if attempt < max_retries:
                        continue
                else:
                    raise RuntimeError(f"Embedding API 返回异常 (HTTP {status})")
            except httpx.ConnectError:
                last_error = "无法连接通义千问 Embedding 服务，请检查网络"
                if attempt < max_retries:
                    continue
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Embedding 返回数据格式异常: {e}")

        raise RuntimeError(last_error or "Embedding 调用失败，请稍后重试")

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
