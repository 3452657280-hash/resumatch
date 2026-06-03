"""应用配置 —— 统一管理 LLM、Embedding 和环境变量。"""
import json
from pathlib import Path
from typing import Any, Iterator

import httpx
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
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


# ── 自定义 LangChain 兼容的 DashScope LLM ──

class DashScopeChatModel(BaseChatModel):
    """通过 DashScope 兼容接口调用通义千问的 LangChain 聊天模型。"""

    model: str = "qwen-max"
    dashscope_api_key: str = ""
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str:
        return "dashscope-qwen"

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
        }
        result = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                role = msg.role
            else:
                role = role_map.get(msg.type, "user")
            result.append({"role": role, "content": msg.content})
        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            **({"stop": stop} if stop else {}),
        }
        resp = httpx.post(
            f"{DASHSCOPE_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]["content"]
        generation = ChatGeneration(message=AIMessage(content=message))
        return ChatResult(generations=[generation])

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}


# ── 自定义 LangChain Embeddings ──

class DashScopeEmbeddings(Embeddings):
    """通过 DashScope 兼容接口调用通义千问 Embedding。"""

    def __init__(self, model: str = "text-embedding-v3", dashscope_api_key: str = ""):
        self.model = model
        self.dashscope_api_key = dashscope_api_key

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        # DashScope 兼容 OpenAI 格式
        payload = {
            "model": self.model,
            "input": text,
        }
        resp = httpx.post(
            f"{DASHSCOPE_BASE}/embeddings",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


def get_llm(temperature: float = 0.1) -> DashScopeChatModel:
    """获取通义千问 LLM 实例。"""
    return DashScopeChatModel(
        model=settings.llm_model,
        dashscope_api_key=settings.dashscope_api_key,
        temperature=temperature,
    )


def get_embeddings() -> DashScopeEmbeddings:
    """获取通义千问 Embedding 实例。"""
    return DashScopeEmbeddings(
        model=settings.embedding_model,
        dashscope_api_key=settings.dashscope_api_key,
    )
