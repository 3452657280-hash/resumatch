"""Agent 简历匹配服务 —— 三阶段式：关键词提取 → 并发搜索 → Rerank → 评分 + 自检。"""
import json
import re

from app.config import DashScopeLLM
from app.models import MatchResult, SkillGap
from services.rag import ResumeRAG


RERANK_PROMPT = """你是一个简历片段重排序专家。给定一个职位描述（JD）和多个简历文本片段，请评估每个片段与 JD 的相关性。

请直接输出 JSON 数组，不要包含其他文字：
[
  {"index": 0, "relevance_score": 0-10的整数, "reason": "简短理由"},
  {"index": 1, "relevance_score": 0-10的整数, "reason": "简短理由"},
  ...
]

评分标准：
- 9-10: 直接命中 JD 中的核心技能或经验要求
- 7-8: 与 JD 要求高度相关，但不完全匹配
- 5-6: 部分相关，涉及 JD 中的某些方面
- 3-4: 弱相关，仅有少量关联
- 1-2: 几乎不相关
- 0: 完全不相关或无关内容

注意：诚实评分，不要给所有片段都打高分。"""

STRUCTURED_KEYWORD_PROMPT = """你是一位简历筛选 Agent。你的任务是分析职位描述（JD），提取出评估候选人时最需要关注的关键词，并按类别分类。

同时，你需要为每个关键词建议一个**最适合搜索的简历段落**。简历常见的段落有：工作经验、项目经历、专业技能、教育背景、自我评价等。

请直接输出 JSON，不要包含其他文字：
{
  "硬性条件": ["Python", "Docker", ...],
  "软性条件": ["团队协作", "沟通能力", ...],
  "业务领域": ["金融", "电商", ...],
  "段落建议": {
    "Python": "专业技能",
    "Docker": "工作经验",
    "沟通能力": "自我评价",
    "金融": "项目经历"
  }
}

分类说明：
- 硬性条件：JD 中明确要求的必备技能或经验（如具体的编程语言、框架、工具），这些是「必须匹配」的
- 软性条件：JD 中提到的软技能或综合素质要求（如沟通能力、团队管理、解决问题的能力）
- 业务领域：JD 所在的行业或业务领域知识（如金融风控、电商推荐、自动驾驶）

段落建议规则：
- 技术关键词（Python、Docker、K8s 等）→ "专业技能" 或 "工作经验"
- 软技能关键词（沟通、团队、管理）→ "自我评价"
- 行业关键词（金融、电商、医疗）→ "项目经历" 或 "工作经验"
- 教育关键词（学历、专业、学校）→ "教育背景"

每个关键词应该是一个具体的技术名词或能力标签。不要输出抽象概念，要输出可在简历中搜索的具体词汇。"""

SCORE_PROMPT = """你是一位资深的简历筛选专家和招聘顾问。你的任务是根据职位描述（JD）和候选人的简历内容进行深度匹配分析。

以下是针对该简历按多个关键词搜索到的内容片段，请综合分析后给出评分。

关键词分为三类，注意它们的权重不同：
- 🔴 硬性条件（必须匹配）— 权重最大，缺少会大幅扣分
- 🟡 软性条件（加分项）— 有则加分，没有不扣分
- 🔵 业务领域（背景要求）— 考察行业经验匹配度

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
  "self_check": "对这个评分的自我评估：硬性条件是否都已覆盖？有无矛盾或遗漏？如果发现问题请降低评分。"
}

评分维度：
- 技术匹配度（40%）— 硬性条件
- 经验相关度（30%）— 业务领域
- 项目/成果质量（20%）
- 软技能与文化适配（10%）— 软性条件

注意：请诚实评分。如果简历中明显缺少 JD 要求的硬性条件，不要给高分。"""


class AgentMatcher:
    """Agent 简历匹配器：三阶段式 —— 关键词提取 → 并发搜索 → Rerank → 评分。"""

    def __init__(self, llm: DashScopeLLM, rag: ResumeRAG):
        self.llm = llm
        self.rag = rag

    def match_single(self, resume_id: str, filename: str, jd_text: str) -> MatchResult:
        """对一份简历进行三阶段 Agent 匹配分析。"""
        resume_text = self.rag.get_resume_text(resume_id)
        if not resume_text or not resume_text.strip():
            return MatchResult(
                resume_id=resume_id, resume_filename=filename,
                overall_score=0,
                summary="未能在向量库中找到该简历的有效文本，请确认简历已成功上传并解析。",
            )

        # ── Phase 1: 提取结构化关键词（按硬性条件 / 软性条件 / 业务领域分类）──
        structured_keywords = self._extract_keywords(jd_text)

        # ── Phase 2: 分类并发搜索 ──
        search_results = self._search_all_keywords(resume_id, structured_keywords)

        # ── Phase 2.5: Rerank 重排序 ──
        reranked_results = self._rerank(jd_text, search_results)

        # ── Phase 3: 评分 + 自检（传入结构化关键词供参考）──
        return self._score(resume_id, filename, jd_text, reranked_results, structured_keywords)

    def _extract_keywords(self, jd_text: str) -> dict:
        """调用 LLM 提取 JD 结构化关键词，按「硬性条件/软性条件/业务领域」分类 + 段落建议。"""
        response = self.llm.invoke([
            {"role": "system", "content": STRUCTURED_KEYWORD_PROMPT},
            {"role": "user", "content": f"职位描述：\n{jd_text}"},
        ])
        try:
            keywords = json.loads(response.strip())
            if isinstance(keywords, dict):
                for cat in ["硬性条件", "软性条件", "业务领域"]:
                    if cat not in keywords:
                        keywords[cat] = []
                if "段落建议" not in keywords:
                    keywords["段落建议"] = {}
                return keywords
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback
        return {"硬性条件": [jd_text[:50]], "软性条件": [], "业务领域": [], "段落建议": {}}

    def _search_all_keywords(self, resume_id: str, structured_keywords: dict) -> list[dict]:
        """对所有关键词分类搜索。

        两步走：
        1. 先用段落建议在指定段落内搜（利用 section 元数据过滤）
        2. 如果结果不够 k 个，再搜全文补满
        """
        seen = set()
        results = []
        section_hints = structured_keywords.get("段落建议", {})

        def add_result(d, keyword, category):
            content = d["content"][:300]
            if content not in seen:
                seen.add(content)
                skills_meta = d["metadata"].get("skills", "")
                direct_hit = keyword.lower() in skills_meta.lower() if skills_meta else False
                results.append({
                    "keyword": keyword,
                    "category": category,
                    "content": content,
                    "direct_skill_match": direct_hit,
                    "section": d["metadata"].get("section", ""),
                })

        search_config = [
            ("硬性条件", 3),   # 硬性条件搜 3 个
            ("软性条件", 1),   # 软性条件搜 1 个
            ("业务领域", 2),   # 业务领域搜 2 个
        ]

        for cat, k in search_config:
            for keyword in structured_keywords.get(cat, []):
                section_name = section_hints.get(keyword, "")
                section_results = []

                # 第 1 步：在建议段落内优先搜
                if section_name:
                    section_results = self.rag.search_by_section(resume_id, keyword, section_name, k=k)

                # 第 2 步：如果段落内搜不够 k 个，用全文搜补齐
                full_results = []
                if len(section_results) < k:
                    extra_needed = k - len(section_results)
                    full_results = self.rag.search_by_keyword(resume_id, keyword, k=extra_needed)

                # 合并结果，段落内优先排在前面
                for d in section_results + full_results:
                    add_result(d, keyword, cat)

        return results

    def _rerank(self, jd_text: str, search_results: list[dict]) -> list[dict]:
        """用 LLM 对检索结果按与 JD 的相关性进行重排序，保留 Top-5。"""
        if len(search_results) <= 3:
            return search_results  # 太少不需要 rerank

        # 构建重排序 Prompt
        items = []
        for i, r in enumerate(search_results):
            items.append(f"[{i}] 关键词: {r['keyword']}\n内容: {r['content']}")

        items_text = "\n\n---\n\n".join(items)
        user_message = (
            f"## 职位描述（JD）\n{jd_text}\n\n"
            f"## 待排序的简历片段\n{items_text}\n\n"
            f"请评估每个片段与 JD 的相关性并给出评分。"
        )

        response = self.llm.invoke([
            {"role": "system", "content": RERANK_PROMPT},
            {"role": "user", "content": user_message},
        ])

        try:
            scores = json.loads(response.strip())
            if isinstance(scores, list) and len(scores) >= 2:
                # 按相关性得分降序排列
                scores.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                # 取 Top-5
                top_indices = [s["index"] for s in scores[:5] if 0 <= s["index"] < len(search_results)]
                reranked = [search_results[i] for i in top_indices]
                return reranked if reranked else search_results[:5]
        except (json.JSONDecodeError, TypeError, IndexError, KeyError):
            pass

        # Fallback: 直接取前 5 个
        return search_results[:5]

    def _score(self, resume_id: str, filename: str, jd_text: str, search_results: list[dict], structured_keywords: dict | None = None) -> MatchResult:
        """基于搜索到的简历内容进行评分（含结构化关键词分类信息）。"""
        context_parts = []
        for r in search_results:
            tag = f"[{r['category']}] " if r.get('category') else ""
            skill_tag = " [技能直接命中]" if r.get('direct_skill_match') else ""
            section_tag = f" (来自: {r['section']})" if r.get('section') else ""
            context_parts.append(f"{tag}{r['content']}{skill_tag}{section_tag}")

        resume_context = "\n\n---\n\n".join(context_parts) if context_parts else "（未找到相关简历内容）"

        # 构建关键词分类摘要
        kw_parts = []
        if structured_keywords:
            if structured_keywords.get("硬性条件"):
                kw_parts.append(f"🔴 硬性条件（必须匹配）：{', '.join(structured_keywords['硬性条件'])}")
            if structured_keywords.get("软性条件"):
                kw_parts.append(f"🟡 软性条件（加分项）：{', '.join(structured_keywords['软性条件'])}")
            if structured_keywords.get("业务领域"):
                kw_parts.append(f"🔵 业务领域（背景要求）：{', '.join(structured_keywords['业务领域'])}")
        kw_summary = "\n".join(kw_parts) if kw_parts else ""

        user_message = (
            f"## 职位描述（JD）\n{jd_text}\n\n"
            f"## 关键词分类\n{kw_summary}\n\n"
            f"## 候选人简历检索内容\n{resume_context}\n\n"
            f"请根据 JD 对该候选人进行匹配分析，特别注意硬性条件的匹配程度。"
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
