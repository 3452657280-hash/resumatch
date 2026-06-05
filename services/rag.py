"""RAG 服务 —— 简历向量化存储与检索（纯 chromadb 实现）。"""
import hashlib
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import DashScopeEmbedder


def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """中文友好的文本分块。"""
    separators = ["\n\n", "\n", "。", ".", " ", ""]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to break at a separator
            best = -1
            for sep in separators:
                if not sep:
                    break
                pos = text.rfind(sep, start, end)
                if pos > best:
                    best = pos
            if best > start:
                end = best + len(separators[0]) if separators[0] and text[best:best+len(separators[0])] == separators[0] else best
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks if chunks else [text.strip()]


class ResumeRAG:
    """简历 RAG 引擎：Chunk → Embedding → ChromaDB 存储 & 检索。"""

    def __init__(self, embedder: DashScopeEmbedder, persist_dir: str):
        self.embedder = embedder
        self.persist_dir = persist_dir
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="resumes",
            metadata={"hnsw:space": "cosine"},
        )

    def add_resume(self, resume_id: str, filename: str, text: str, content_hash: str = "") -> int:
        """将简历文本分块后存入向量库。返回 chunk 数量。"""
        chunks = split_text(text)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_many(chunks)
        ids = [f"{resume_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"resume_id": resume_id, "filename": filename, "chunk": i, "content_hash": content_hash}
            for i in range(len(chunks))
        ]
        self._collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """检索与 query 最相关的简历片段。"""
        q_emb = self.embedder.embed(query)
        results = self._collection.query(query_embeddings=[q_emb], n_results=k)
        return self._format_results(results)

    def search_by_resume_id(self, resume_id: str, query: str, k: int = 3) -> list[dict]:
        """检索指定简历中与 query 相关的片段。"""
        q_emb = self.embedder.embed(query)
        results = self._collection.query(query_embeddings=[q_emb], n_results=k * 5)
        docs = self._format_results(results)
        return [d for d in docs if d["metadata"].get("resume_id") == resume_id][:k]

    def search_by_keyword(self, resume_id: str, keyword: str, k: int = 2) -> list[dict]:
        """使用特定关键词检索（Agent 搜索工具）。"""
        return self.search_by_resume_id(resume_id, keyword, k=k)

    def _format_results(self, results) -> list[dict]:
        formatted = []
        if not results["ids"] or not results["ids"][0]:
            return formatted
        for i, doc_id in enumerate(results["ids"][0]):
            formatted.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })
        return formatted

    def get_all_resumes(self) -> list[dict]:
        """获取向量库中所有简历信息（去重）。"""
        all_data = self._collection.get()
        seen = {}
        for meta in (all_data.get("metadatas") or []):
            if meta and (rid := meta.get("resume_id")):
                if rid not in seen:
                    seen[rid] = {"resume_id": rid, "filename": meta.get("filename", "")}
        return list(seen.values())

    def get_resume_text(self, resume_id: str) -> str | None:
        """获取指定简历的原始文本。"""
        all_data = self._collection.get()
        texts = []
        for meta, doc in zip(all_data.get("metadatas") or [], all_data.get("documents") or []):
            if meta and meta.get("resume_id") == resume_id and doc:
                texts.append(doc)
        return "\n---\n".join(texts) if texts else None

    def exists_by_content_hash(self, content_hash: str) -> bool:
        all_data = self._collection.get()
        for meta in (all_data.get("metadatas") or []):
            if meta and meta.get("content_hash") == content_hash:
                return True
        return False

    def delete_resume(self, resume_id: str) -> bool:
        all_data = self._collection.get()
        ids_to_delete = []
        for meta, doc_id in zip(all_data.get("metadatas") or [], all_data.get("ids") or []):
            if meta and meta.get("resume_id") == resume_id:
                ids_to_delete.append(doc_id)
        if not ids_to_delete:
            return False
        self._collection.delete(ids=ids_to_delete)
        return True

    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()
