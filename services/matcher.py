"""Agent 简历匹配服务 —— 四阶段：JD 标签提取 → 并发搜索 → RAG Rerank → LLM 评分 + 独立自检。"""
import json
import re

from app.config import DashScopeLLM
from app.models import MatchResult, SkillGap
from services.rag import ResumeRAG


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
}

评分维度：
- 技术匹配度（40%）— 硬性条件
- 经验相关度（30%）— 业务领域
- 项目/成果质量（20%）
- 软技能与文化适配（10%）— 软性条件

⚠️ 重要规则（必须遵守）：
1. **用项目经验证明能力** — 如果候选人做过 AI 智能体、RAG 系统、大模型部署等项目，就应视为具备大模型 API 使用经验，即使简历中没有写出"OpenAI""DeepSeek"等具体名词。项目本身就是最好的证据。
2. **不要写"没有直接提到XX"这种劣势** — 劣势必须是有实际业务影响的能力缺失。如果简历通过项目经历展示了相关能力，就不算劣势。
3. **skill_gaps 只列真实缺失** — 只有在简历的任何项目中都找不到相关实践证据时，才标记为 missing 或 weak。
4. **客观诚实** — 如果简历内容满足 JD 要求就给高分，不故意压分。"""


SELF_CHECK_PROMPT = """你是一位严格的简历评分审核官。你的任务是对上一轮简历匹配的评分结果进行复核。

请对比 JD 要求的硬性条件与简历实际内容，逐项检查：

1. JD 要求的每个硬性条件，简历是否真的具备？如果存在缺失但评分偏高，**必须建议降分**。
2. 评分是否过高或过低？如果发现不合理，请调整到合理的分数。
3. 是否存在 LLM 幻觉——简历中没有提到的技能，却被标记为 "matched"？

请输出以下 JSON：

{
  "pass": true/false,
  "adjusted_score": 0-100的整数（如果原分合理则保持不变）,
  "issues": ["发现的问题1", "发现的问题2", ...],
  "final_verdict": "最终审核结论（一句话）"
}

注意：
- 如果发现硬性条件缺失但原评分较高，请诚实降分。
- 如果原评分合理，就保持分数不变，pass 为 true。
- 你是最终审核官，你的分数是最终分数。
- **不要编造没有意义的劣势**。候选人通过项目经验展示了相关能力，就应视为具备该能力。
- **特别提醒**：做过 AI 智能体、RAG、大模型部署等项目的候选人，100% 具备大模型 API 使用经验，不得将其列为缺失项。"""


def _group_by_keyword(items: list[dict]) -> list[list[dict]]:
    """将搜索结果按 keyword 分组。"""
    groups = {}
    for item in items:
        kw = item.get("keyword", "")
        groups.setdefault(kw, []).append(item)
    return list(groups.values())


class AgentMatcher:
    """Agent 简历匹配器：四阶段 —— JD 标签提取 → 并发搜索 → RAG Rerank → LLM 评分 + 独立自检。"""

    def __init__(self, llm: DashScopeLLM, rag: ResumeRAG):
        self.llm = llm
        self.rag = rag

    def match_single(self, resume_id: str, filename: str, jd_text: str) -> MatchResult:
        """对一份简历进行四阶段 Agent 匹配分析。"""
        resume_text = self.rag.get_resume_text(resume_id)
        if not resume_text or not resume_text.strip():
            return MatchResult(
                resume_id=resume_id, resume_filename=filename,
                overall_score=0,
                summary="未能在向量库中找到该简历的有效文本，请确认简历已成功上传并解析。",
            )

        # ── Phase 1: 提取结构化关键词 ──
        try:
            structured_keywords = self._extract_keywords(jd_text)
        except RuntimeError as e:
            return MatchResult(
                resume_id=resume_id, resume_filename=filename,
                overall_score=0,
                summary=f"JD 关键词提取失败：{e}",
            )

        # ── Phase 2: 分类并发搜索 ──
        try:
            grouped_results = self._search_all_keywords(resume_id, structured_keywords)
        except RuntimeError as e:
            return MatchResult(
                resume_id=resume_id, resume_filename=filename,
                overall_score=0,
                summary=f"简历检索失败：{e}",
            )

        all_results = grouped_results["硬性条件"] + grouped_results["软性条件"] + grouped_results["业务领域"]
        if not all_results:
            return MatchResult(
                resume_id=resume_id, resume_filename=filename,
                overall_score=0,
                summary="在向量库中未找到与 JD 相关的简历内容，请确认简历已正确解析并包含相关经验。",
            )

        # ── Phase 2.5: 硬性条件不做重排，全部保留；其他做重排 ──
        #  硬性条件是匹配核心，每个关键词搜到的结果直接保留 top 2，不做重排（避免误杀证据）
        #  业务领域和软性条件各取 top 1
        reranked_results = []
        try:
            # 硬性条件：不做重排，直接取每个关键词的前 2 条
            for kw_items in _group_by_keyword(grouped_results.get("硬性条件", [])):
                reranked_results.extend(kw_items[:2])

            # 业务领域：重排选 top 1
            for kw_items in _group_by_keyword(grouped_results.get("业务领域", [])):
                if len(kw_items) <= 1:
                    reranked_results.extend(kw_items)
                else:
                    reranked_results.extend(self.rag._rerank(kw_items, jd_text, top_k=1))

            # 软性条件：重排选 top 1
            for kw_items in _group_by_keyword(grouped_results.get("软性条件", [])):
                if len(kw_items) <= 1:
                    reranked_results.extend(kw_items)
                else:
                    reranked_results.extend(self.rag._rerank(kw_items, jd_text, top_k=1))

            # 最多 15 条，防止 LLM prompt 过长
            if len(reranked_results) > 15:
                reranked_results = reranked_results[:15]
        except RuntimeError:
            reranked_results = []
            for cat in ("硬性条件", "业务领域", "软性条件"):
                reranked_results.extend(grouped_results.get(cat, [])[:3])

        # ── Phase 3: 评分 ──
        try:
            result = self._score(resume_id, filename, jd_text, reranked_results, structured_keywords)
        except RuntimeError as e:
            return MatchResult(
                resume_id=resume_id, resume_filename=filename,
                overall_score=0,
                summary=f"简历评分失败：{e}",
            )

        # ── Phase 4: 二次审查（独立 LLM 复核） ──
        try:
            result = self._self_check(resume_id, filename, jd_text, structured_keywords, result)
        except RuntimeError:
            # 自检失败降级，使用原评分结果
            pass

        return result

    def match_single_stream(self, resume_id: str, filename: str, jd_text: str):
        """对一份简历进行匹配分析，以生成器方式逐阶段推送进度。

        Yields:
            dict: {"type": "progress", "phase": str, "current": int, "total": int, "label": str}
            或 {"type": "result", "data": dict} 或 {"type": "error", "message": str}
        """
        resume_text = self.rag.get_resume_text(resume_id)
        if not resume_text or not resume_text.strip():
            yield {"type": "result", "data": MatchResult(
                resume_id=resume_id, resume_filename=filename,
                overall_score=0,
                summary="未能在向量库中找到该简历的有效文本，请确认简历已成功上传并解析。",
            ).model_dump()}
            return

        phases = [
            ("extract", "🔍 提取关键词"),
            ("search", "📚 搜索文本块"),
            ("rerank", "⚖️ Rerank 重排"),
            ("score", "⚖️ 评分自检"),
        ]
        total = len(phases)

        # ── Phase 1: 提取结构化关键词 ──
        yield {"type": "progress", "phase": "extract", "current": 1, "total": total, "label": "🔍 提取关键词"}
        try:
            structured_keywords = self._extract_keywords(jd_text)
        except RuntimeError as e:
            yield {"type": "error", "message": f"JD 关键词提取失败：{e}"}
            return

        # ── Phase 2: 分类并发搜索 ──
        yield {"type": "progress", "phase": "search", "current": 2, "total": total, "label": "📚 搜索文本块"}
        try:
            grouped_results = self._search_all_keywords(resume_id, structured_keywords)
        except RuntimeError as e:
            yield {"type": "error", "message": f"简历检索失败：{e}"}
            return

        all_results = grouped_results["硬性条件"] + grouped_results["软性条件"] + grouped_results["业务领域"]
        if not all_results:
            yield {"type": "result", "data": MatchResult(
                resume_id=resume_id, resume_filename=filename,
                overall_score=0,
                summary="在向量库中未找到与 JD 相关的简历内容。",
            ).model_dump()}
            return

        # ── Phase 2.5: Rerank（按关键词分别重排，保证每个条件都有证据）──
        yield {"type": "progress", "phase": "rerank", "current": 3, "total": total, "label": "⚖️ Rerank 重排"}
        reranked_results = []
        try:
            for kw_items in _group_by_keyword(grouped_results.get("硬性条件", [])):
                reranked_results.extend(kw_items[:2])
            for kw_items in _group_by_keyword(grouped_results.get("业务领域", [])):
                if len(kw_items) <= 1:
                    reranked_results.extend(kw_items)
                else:
                    reranked_results.extend(self.rag._rerank(kw_items, jd_text, top_k=1))
            for kw_items in _group_by_keyword(grouped_results.get("软性条件", [])):
                if len(kw_items) <= 1:
                    reranked_results.extend(kw_items)
                else:
                    reranked_results.extend(self.rag._rerank(kw_items, jd_text, top_k=1))
            if len(reranked_results) > 15:
                reranked_results = reranked_results[:15]
        except RuntimeError:
            reranked_results = []
            for cat in ("硬性条件", "业务领域", "软性条件"):
                reranked_results.extend(grouped_results.get(cat, [])[:3])

        # ── Phase 3: 评分 ──
        yield {"type": "progress", "phase": "score", "current": 4, "total": total, "label": "⚖️ 评分自检"}
        try:
            result = self._score(resume_id, filename, jd_text, reranked_results, structured_keywords)
        except RuntimeError as e:
            yield {"type": "error", "message": f"简历评分失败：{e}"}
            return

        # ── Phase 4: 二次审查 ──
        try:
            result = self._self_check(resume_id, filename, jd_text, structured_keywords, result)
        except RuntimeError:
            pass

        yield {"type": "result", "data": result.model_dump()}

    def _extract_keywords(self, jd_text: str) -> dict:
        """调用 LLM 提取 JD 结构化关键词，按「硬性条件/软性条件/业务领域」分类 + 段落建议。"""
        try:
            response = self.llm.invoke([
                {"role": "system", "content": STRUCTURED_KEYWORD_PROMPT},
                {"role": "user", "content": f"职位描述：\n{jd_text}"},
            ])
            keywords = json.loads(response.strip())
            if isinstance(keywords, dict):
                for cat in ["硬性条件", "软性条件", "业务领域"]:
                    if cat not in keywords:
                        keywords[cat] = []
                if "段落建议" not in keywords:
                    keywords["段落建议"] = {}
                return keywords
        except (json.JSONDecodeError, TypeError, RuntimeError):
            pass
        # Fallback
        return {"硬性条件": [jd_text[:50]], "软性条件": [], "业务领域": [], "段落建议": {}}

    def _search_all_keywords(self, resume_id: str, structured_keywords: dict) -> dict[str, list[dict]]:
        """对所有关键词分类搜索，返回按类别分组的结果。"""
        seen = set()
        results = {"硬性条件": [], "软性条件": [], "业务领域": []}
        section_hints = structured_keywords.get("段落建议", {})

        def add_result(d, keyword, category):
            content = d["content"][:300]
            if content not in seen:
                seen.add(content)
                skills_meta = d["metadata"].get("skills", "")
                direct_hit = keyword.lower() in skills_meta.lower() if skills_meta else False
                results[cat].append({
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



    def _score(self, resume_id: str, filename: str, jd_text: str, search_results: list[dict], structured_keywords: dict | None = None) -> MatchResult:
        """基于搜索到的简历内容进行评分（含结构化关键词分类信息）。"""
        context_parts = []
        for r in search_results:
            tag = f"[{r.get('category', '')}] " if r.get('category') else ""
            skill_tag = " [技能直接命中]" if r.get('direct_skill_match') else ""
            section_tag = f" (来自: {r.get('section', '')})" if r.get('section') else ""
            content_preview = (r.get('content') or '')[:500]
            context_parts.append(f"{tag}{content_preview}{skill_tag}{section_tag}")

        resume_context = "\n\n---\n\n".join(context_parts) if context_parts else "（未找到相关简历内容）"

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

        try:
            response = self.llm.invoke([
                {"role": "system", "content": SCORE_PROMPT},
                {"role": "user", "content": user_message},
            ])
        except RuntimeError as e:
            raise RuntimeError(f"评分 LLM 调用失败: {e}")

        parsed = self._parse_json(response)
        if parsed is None:
            raise RuntimeError("评分 LLM 返回格式异常，无法解析评分结果")

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

    def _self_check(self, resume_id: str, filename: str, jd_text: str, structured_keywords: dict | None, initial_result: MatchResult) -> MatchResult:
        """独立调 LLM 对评分结果进行二次审查，如有问题则调整分数。"""
        keywords_info = ""
        if structured_keywords:
            for cat in ("硬性条件", "软性条件", "业务领域"):
                vals = structured_keywords.get(cat, [])
                if vals:
                    keywords_info += f"- {cat}: {', '.join(vals)}\n"

        user_message = (
            f"## 职位描述（JD）\n{jd_text}\n\n"
            f"## 提取的关键词\n{keywords_info}\n\n"
            f"## 初审评分结果\n"
            f"总分: {initial_result.overall_score}\n"
            f"总结: {initial_result.summary}\n"
            f"优势: {', '.join(initial_result.strengths)}\n"
            f"劣势: {', '.join(initial_result.weaknesses)}\n"
            f"技能匹配: {json.dumps([s.model_dump() for s in initial_result.skill_gaps], ensure_ascii=False)}\n\n"
            f"请复核上述评分是否合理，如有问题请调整。"
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": SELF_CHECK_PROMPT},
                {"role": "user", "content": user_message},
            ])
        except RuntimeError:
            # 自检 LLM 调用失败，降级返回原结果
            return initial_result

        try:
            check_result = json.loads(response.strip())
            adjusted_score = check_result.get("adjusted_score", initial_result.overall_score)
            issues = check_result.get("issues", [])

            if adjusted_score != initial_result.overall_score or not check_result.get("pass", True):
                original_summary = initial_result.summary
                issues_text = "；".join(issues) if issues else "评分已调整"
                new_summary = f"{original_summary}（经二次审查：{check_result.get('final_verdict', issues_text)}）"
                return MatchResult(
                    resume_id=resume_id,
                    resume_filename=filename,
                    overall_score=adjusted_score,
                    summary=new_summary,
                    strengths=initial_result.strengths,
                    weaknesses=initial_result.weaknesses,
                    skill_gaps=initial_result.skill_gaps,
                    suggestions=initial_result.suggestions,
                )
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

        return initial_result

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
