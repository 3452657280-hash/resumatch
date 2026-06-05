"""Agent 简历匹配服务 —— 两阶段式：关键词提取 → 并发搜索 → 评分 + 自检。"""
import json
import re

from app.config import DashScopeLLM
from app.models import MatchResult, SkillGap
from services.rag import ResumeRAG

KEYWORD_EXTRACT_PROMPT = """你是一位简历筛选 Agent。你的任务是分析职位描述（JD），提取出评估候选人时最需要关注的 3-5 个关键词。

请直接输出 JSON 列表，不要包含其他文字：
["关键词1", "关键词2", "关键词3", ...]

每个关键词应该是一个具体的技术名词或能力标签，例如 "Python"、"Docker"、"项目管理"、"Kubernetes" 等。
关键词要覆盖技术栈、业务领域、软技能三个维度。
不要输出抽象概念，要输出可直接在简历中搜索的具体词汇。"""

SCORE_PROMPT = """你是一位资深的简历筛选专家和招聘顾问。你的任务是根据职位描述（JD）和候选人的简历内容进行深度匹配分析。

以下是针对该简历按多个关键词搜索到的内容片段，请综合分析后给出评分。

请严格按照以下 JSON 格式输出，不要包含其他文字：

{
  "overall_score": 0-100的整数,
  "summary": "2-3句总体评价",
  "strengths": ["优势1", "优势2", ...],
  "weaknesses": ["劣势1", "劣势2", ...],
  "skill_gaps": [
    {"skill": "技能名", "status": "matched|missing|weak", "suggestion": "提升建议"}
  ],
  "suggestions": ["改进建议1", "改进建议2", ...],
  "self_check": "对这个评分的自我评估，检查是否有遗漏或矛盾。如果发现问题，请降低评分。"
}

评分维度：
- 技术匹配度（40%）
- 经验相关度（30%）
- 项目/成果质量（20%）
- 软技能与文化适配（10%）

注意：请诚实评分。如果简历中明显缺少 JD 要求的关键技能，不要给高分。"""


class AgentMatcher:
    """Agent 简历匹配器：两阶段式 —— 关键词提取 → 并发搜索 → 评分。"""

    def __init__(self, llm: DashScopeLLM, rag: ResumeRAG):
        self.llm = llm
        self.rag = rag

    def match_single(self, resume_id: str, filename: str, jd_text: str) -> MatchResult:
        """对一份简历进行两阶段 Agent 匹配分析。"""
        resume_text = self.rag.get_resume_text(resume_id)
        if not resume_text or not resume_text.strip():
            return MatchResult(
                resume_id=resume_id, resume_filename=filename,
                overall_score=0,
                summary="未能在向量库中找到该简历的有效文本，请确认简历已成功上传并解析。",
            )

        # ── Phase 1: 提取搜索关键词 ──
        keywords = self._extract_keywords(jd_text)

        # ── Phase 2: 并发搜索 ──
        search_results = self._search_all_keywords(resume_id, keywords)

        # ── Phase 3: 评分 + 自检 ──
        return self._score(resume_id, filename, jd_text, search_results)

    def _extract_keywords(self, jd_text: str) -> list[str]:
        """调用 LLM 提取 JD 核心关键词。"""
        response = self.llm.invoke([
            {"role": "system", "content": KEYWORD_EXTRACT_PROMPT},
            {"role": "user", "content": f"职位描述：\n{jd_text}"},
        ])
        try:
            keywords = json.loads(response.strip())
            if isinstance(keywords, list) and len(keywords) >= 2:
                return keywords[:6]
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: 用分隔符拆分
        for sep in ["\n", ","]:
            if sep in response:
                return [k.strip().strip("-* ") for k in response.split(sep) if k.strip()][:6]
        return [jd_text[:50]]

    def _search_all_keywords(self, resume_id: str, keywords: list[str]) -> list[dict]:
        """对所有关键词进行搜索，去重合并结果。"""
        seen = set()
        results = []
        for keyword in keywords:
            docs = self.rag.search_by_keyword(resume_id, keyword, k=2)
            for d in docs:
                content = d["content"][:300]
                if content not in seen:
                    seen.add(content)
                    results.append({"keyword": keyword, "content": content})
        return results

    def _score(self, resume_id: str, filename: str, jd_text: str, search_results: list[dict]) -> MatchResult:
        """基于搜索到的简历内容进行评分。"""
        context_parts = []
        for r in search_results:
            context_parts.append(f"[关键词: {r['keyword']}]\n{r['content']}")

        resume_context = "\n\n---\n\n".join(context_parts) if context_parts else "（未找到相关简历内容）"

        user_message = (
            f"## 职位描述（JD）\n{jd_text}\n\n"
            f"## 候选人简历检索内容\n{resume_context}\n\n"
            f"请根据 JD 对该候选人进行匹配分析。"
        )

        response = self.llm.invoke([
            {"role": "system", "content": SCORE_PROMPT},
            {"role": "user", "content": user_message},
        ])

        parsed = self._parse_json(response)
        if parsed is None:
            return MatchResult(
                resume_id=resume_id, resume_filename=filename,
                overall_score=0, summary="分析结果解析失败",
            )

        skill_gaps = []
        for sg in parsed.get("skill_gaps") or []:
            if isinstance(sg, dict):
                skill_gaps.append(SkillGap(**sg))

        return MatchResult(
            resume_id=resume_id,
            resume_filename=filename,
            overall_score=parsed.get("overall_score", 0),
            summary=parsed.get("summary", ""),
            strengths=parsed.get("strengths") or [],
            weaknesses=parsed.get("weaknesses") or [],
            skill_gaps=skill_gaps,
            suggestions=parsed.get("suggestions") or [],
        )

    def _parse_json(self, text: str) -> dict | None:
        """从 LLM 输出中提取并解析 JSON。"""
        raw = text.strip()
        # Find outermost JSON
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    json_str = raw[start:i + 1]
                    break
        else:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            json_str = m.group() if m else raw

        # Multi-strategy parse
        strategies = [
            lambda s: json.loads(s, strict=False),
            lambda s: json.loads(re.sub(r',\s*}', '}', re.sub(r',\s*\]', ']', s)), strict=False),
            lambda s: json.loads(s, strict=False),
        ]
        for strategy in strategies:
            try:
                return strategy(json_str)
            except (json.JSONDecodeError, TypeError):
                continue
        return None
