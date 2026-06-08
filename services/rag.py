"""RAG 服务 —— 简历向量化存储与检索（纯 chromadb 实现）。"""
import hashlib
import re
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import DashScopeEmbedder
from services.reranker import RerankerService

# 常见技能关键词列表（用于自动标注 chunk 所属技能）
COMMON_SKILLS = [
    # 编程语言
    "Python", "Java", "JavaScript", "TypeScript", "Go", "Golang",
    "Rust", "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R",
    # 前端
    "React", "Vue", "Vue.js", "Angular", "Node.js", "HTML", "CSS",
    "jQuery", "Webpack", "Vite",
    # 后端框架
    "Django", "Flask", "FastAPI", "Spring Boot", "Spring", "Express",
    "Next.js", "GraphQL", "gRPC",
    # DevOps
    "Docker", "Kubernetes", "K8s", "Jenkins", "CI/CD", "GitHub Actions",
    "GitLab CI", "Ansible", "Terraform", "Nginx", "Linux", "Shell",
    # 云平台
    "AWS", "Azure", "GCP", "阿里云", "腾讯云", "华为云",
    # 数据库 / 中间件
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "Kafka", "RabbitMQ", "SQLite", "Oracle",
    # AI / ML
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    "LangChain", "LLM", "RAG", "OpenAI", "Transformer", "LoRA",
    # 大数据
    "Spark", "Hadoop", "Flink", "Hive", "HBase",
    # 架构
    "微服务", "分布式", "高并发", "高可用", "容器化", "云原生",
    # 管理
    "项目管理", "团队管理", "敏捷开发", "Scrum", "产品管理",
]


def extract_skills_from_text(text: str) -> list[str]:
    """从文本中扫描出包含的技能关键词。

    英文技能用单词边界匹配（避免 "R" 匹配到 "Docker"），
    中文技能直接用子串匹配（中文天然自分隔）。
    """
    found = set()
    for skill in COMMON_SKILLS:
        if re.search(r'[一-鿿]', skill):
            # 中文技能：直接子串匹配
            if skill in text:
                found.add(skill)
        else:
            # 英文技能：ASCII 单词边界匹配，防止 "R" 匹配 "Docker" 中的 r
            pattern = re.compile(
                r'(?<![a-zA-Z])' + re.escape(skill) + r'(?![a-zA-Z])',
                re.IGNORECASE,
            )
            if pattern.search(text):
                found.add(skill)
    return sorted(found)


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
        self._reranker = RerankerService()
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="resumes",
            metadata={"hnsw:space": "cosine"},
        )

    def add_resume(self, resume_id: str, filename: str, text: str, content_hash: str = "", sections: list[dict] | None = None) -> int:
        """将简历分块后存入向量库。支持按语义段落分块（含 section / skills 元数据）。"""
        if sections:
            # ── 按段落分别分块 ──
            all_chunks = []
            all_metadatas = []
            for sec in sections:
                section_name = sec["section"]
                section_text = sec["text"]
                chunks = split_text(section_text)
                if not chunks:
                    continue
                section_skills = extract_skills_from_text(section_text)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "resume_id": resume_id,
                        "filename": filename,
                        "section": section_name,
                        "chunk": i,
                        "skills": ",".join(section_skills),
                        "content_hash": content_hash,
                    })
            if not all_chunks:
                return 0
            embeddings = self.embedder.embed_many(all_chunks)
            ids = [f"{resume_id}_{i}" for i in range(len(all_chunks))]
            self._collection.add(ids=ids, embeddings=embeddings, documents=all_chunks, metadatas=all_metadatas)
            return len(all_chunks)

        # ── 无段落信息时回退为全文分块 ──
        chunks = split_text(text)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_many(chunks)
        ids = [f"{resume_id}_{i}" for i in range(len(chunks))]
        skills = extract_skills_from_text(text)
        skills_str = ",".join(skills)
        metadatas = [
            {"resume_id": resume_id, "filename": filename, "section": "全文", "chunk": i, "skills": skills_str, "content_hash": content_hash}
            for i in range(len(chunks))
        ]
        self._collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """检索与 query 最相关的简历片段。"""
        q_emb = self.embedder.embed(query)
        results = self._collection.query(query_embeddings=[q_emb], n_results=k)
        return self._format_results(results)

    def _rerank(self, search_results: list[dict], jd_text: str, top_k: int = 5) -> list[dict]:
        """对搜索结果按 JD 相关性进行 Cross-Encoder 重排。"""
        return self._reranker.rerank_results(jd_text, search_results, top_k=top_k)

    def search_by_resume_id(self, resume_id: str, query: str, k: int = 3) -> list[dict]:
        """检索指定简历中与 query 相关的片段。"""
        q_emb = self.embedder.embed(query)
        results = self._collection.query(query_embeddings=[q_emb], n_results=k * 5)
        docs = self._format_results(results)
        return [d for d in docs if d["metadata"].get("resume_id") == resume_id][:k]

    def search_by_keyword(self, resume_id: str, keyword: str, k: int = 2) -> list[dict]:
        """使用特定关键词检索（Agent 搜索工具）。"""
        return self.search_by_resume_id(resume_id, keyword, k=k)

    def search_by_section(self, resume_id: str, query: str, section: str, k: int = 2) -> list[dict]:
        """在指定段落的范围内检索与 query 相关的片段（利用 ChromaDB where 过滤）。"""
        q_emb = self.embedder.embed(query)
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=k * 3,
            where={"$and": [{"resume_id": resume_id}, {"section": section}]},
        )
        return self._format_results(results)[:k]

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
