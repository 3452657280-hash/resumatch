"""重排服务 —— 基于 Cross-Encoder 的简历片段重排序。

使用 BAAI/bge-reranker-v2-m3 模型对搜索结果按 JD 相关性重排，
替代原来的 LLM 重排，速度快 10-50 倍，零 API 费用。

如 HuggingFace 无法连接，会自动降级为不进行重排（返回原顺序）。
可通过设置环境变量 HF_ENDPOINT=https://hf-mirror.com 使用国内镜像。
"""
import logging
import os

logger = logging.getLogger("resumatch")

# 国内镜像
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class RerankerService:
    """Cross-Encoder 重排器，本地离线运行。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._available = True

    def _load(self):
        """懒加载模型（首次使用才下载）。"""
        if self._model is not None:
            return True
        if not self._available:
            return False
        try:
            logger.info(f"正在加载重排模型: {self.model_name}（首次加载需下载约 1.1GB）")
            from sentence_transformers import CrossEncoder
            import time
            start = time.time()
            self._model = CrossEncoder(self.model_name)
            logger.info(f"重排模型加载完成（耗时 {time.time()-start:.0f}s）")
            return True
        except Exception as e:
            logger.warning(f"重排模型加载失败: {e}，将跳过重排步骤（不影响匹配结果）")
            self._available = False
            return False

    def rerank(self, query: str, candidates: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """对候选文本按与 query 的相关性排序。

        Args:
            query: 查询文本（JD 关键词或 JD 全文）
            candidates: 候选文本列表
            top_k: 返回前 k 个结果的索引和分数

        Returns:
            [(index, score), ...] 按分数降序排列
        """
        if not self._load():
            # 模型不可用，返回原始顺序（假装分数递减）
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k, len(candidates)))]

        if not candidates:
            return []

        pairs = [[query, c] for c in candidates]
        try:
            scores = self._model.predict(pairs)
            indexed = list(enumerate(scores.tolist()))
            indexed.sort(key=lambda x: x[1], reverse=True)
            return indexed[:top_k]
        except Exception as e:
            logger.warning(f"重排预测失败: {e}，跳过重排")
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k, len(candidates)))]

    def rerank_results(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        """对搜索结果列表按与 query 的相关性重排。

        Args:
            query: 查询文本（JD 全文或其关键词）
            results: 搜索结果列表 [{"content": "...", ...}, ...]
            top_k: 返回前 k 个结果

        Returns:
            重排后的结果列表（前 top_k 个）
        """
        if len(results) <= 3:
            return results

        candidates = [r.get("content", "")[:512] for r in results]
        ranked = self.rerank(query, candidates, top_k=min(top_k, len(results)))

        return [results[idx] for idx, _ in ranked]
