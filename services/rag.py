"""RAG 服务 —— 简历向量化存储与检索。"""
import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ResumeRAG:
    """简历 RAG 引擎：Chunk → Embedding → ChromaDB 存储 & 检索。"""

    def __init__(self, embeddings: Embeddings, persist_dir: str):
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        self._vector_store: Chroma | None = None

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        return self._vector_store

    def add_resume(self, resume_id: str, filename: str, text: str, content_hash: str = "") -> int:
        """将简历文本分块后存入向量库。返回 chunk 数量。"""
        chunks = self.splitter.split_text(text)
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "resume_id": resume_id,
                    "filename": filename,
                    "chunk": i,
                    "content_hash": content_hash,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        self.vector_store.add_documents(docs)
        return len(chunks)

    def search(self, query: str, k: int = 5) -> list[Document]:
        """检索与 query 最相关的简历片段。"""
        return self.vector_store.similarity_search(query, k=k)

    def search_by_resume_id(self, resume_id: str, query: str, k: int = 3) -> list[Document]:
        """检索指定简历中与 query 相关的片段。"""
        docs = self.vector_store.similarity_search(query, k=k * 3)
        return [d for d in docs if d.metadata.get("resume_id") == resume_id][:k]

    def get_all_resumes(self) -> list[dict]:
        """获取向量库中所有简历信息（去重），返回 [{resume_id, filename}]。"""
        all_docs = self.vector_store.get()
        seen = {}
        for meta in all_docs.get("metadatas", []):
            if rid := meta.get("resume_id"):
                if rid not in seen:
                    seen[rid] = {
                        "resume_id": rid,
                        "filename": meta.get("filename", ""),
                    }
        return list(seen.values())

    def exists_by_content_hash(self, content_hash: str) -> bool:
        """检查内容哈希是否已存在（防止重复上传）。"""
        all_docs = self.vector_store.get()
        for meta in all_docs.get("metadatas", []):
            if meta.get("content_hash") == content_hash:
                return True
        return False

    def delete_resume(self, resume_id: str) -> bool:
        """删除指定简历的所有向量数据。返回是否删除成功。"""
        all_docs = self.vector_store.get()
        ids_to_delete = [
            mid for mid, meta in zip(all_docs.get("ids", []), all_docs.get("metadatas", []))
            if meta.get("resume_id") == resume_id
        ]
        if not ids_to_delete:
            return False
        self.vector_store.delete(ids=ids_to_delete)
        return True

    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        """计算文件内容的 SHA256 哈希。"""
        return hashlib.sha256(content).hexdigest()
