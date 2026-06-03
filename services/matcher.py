"""简历匹配服务 —— 结合 RAG 检索 + LLM 评分。"""
import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.models import MatchResult, SkillGap
from services.rag import ResumeRAG

SYSTEM_PROMPT = """你是一位资深的简历筛选专家和招聘顾问。你的任务是根据职位描述（JD）对候选人简历进行深度分析。

请严格按照以下 JSON 格式输出分析结果，不要包含任何其他文字：

{
  "reasoning": "简要推理过程",
  "overall_score": 0-100的整数,
  "summary": "2-3句总体评价",
  "strengths": ["优势1", "优势2", ...],
  "weaknesses": ["劣势1", "劣势2", ...],
  "skill_gaps": [
    {"skill": "技能名", "status": "matched|missing|weak", "suggestion": "提升建议"}
  ],
  "suggestions": ["改进建议1", "改进建议2", ...]
}

评分维度：
- 技术匹配度（40%）
- 经验相关度（30%）
- 项目/成果质量（20%）
- 软技能与文化适配（10%）"""


class ResumeMatcher:
    """简历匹配分析器。"""

    def __init__(self, llm: BaseChatModel, rag: ResumeRAG):
        self.llm = llm
        self.rag = rag

    def match_single(self, resume_id: str, filename: str, jd_text: str) -> MatchResult:
        """对一份简历进行匹配分析。"""
        relevant_chunks = self.rag.search_by_resume_id(resume_id, jd_text, k=5)
        resume_context = "\n\n".join(c.page_content for c in relevant_chunks)

        if not resume_context.strip():
            return MatchResult(
                resume_id=resume_id,
                resume_filename=filename,
                overall_score=0,
                summary="未能在向量库中找到该简历的有效文本，请确认简历已成功上传并解析。",
            )

        user_message = (
            f"## 职位描述（JD）\n{jd_text}\n\n"
            f"## 候选人简历内容\n{resume_context}\n\n"
            f"请根据 JD 对该候选人进行匹配分析。"
        )

        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])

        return self._parse_response(response.content, resume_id, filename)

    def _parse_response(
        self, content: str, resume_id: str, filename: str
    ) -> MatchResult:
        """解析 LLM 返回的 JSON 结果。"""
        raw = content.strip()
        json_str = self._extract_json(raw)
        data = self._try_parse(json_str)
        if data is None:
            return MatchResult(
                resume_id=resume_id,
                resume_filename=filename,
                overall_score=0,
                summary="分析结果解析失败，原始输出：" + raw[:300],
            )
        skill_gaps = [SkillGap(**sg) for sg in data.get("skill_gaps", [])]
        return MatchResult(
            resume_id=resume_id,
            resume_filename=filename,
            overall_score=data.get("overall_score", 0),
            summary=data.get("summary", ""),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            skill_gaps=skill_gaps,
            suggestions=data.get("suggestions", []),
        )

    def _extract_json(self, text: str) -> str:
        """从 LLM 输出中提取最外层 JSON 对象。"""
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start:i + 1]
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return m.group() if m else text

    def _try_parse(self, s: str):
        """多策略解析 JSON，处理 LLM 常见的格式问题。"""
        # Clean s before attempts
        s = s.strip()
        
        # Strategy 1: direct parse with strict=False (Python 3.12+)
        try:
            return json.loads(s, strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Strategy 2: remove trailing commas in objects and arrays
        import re
        try:
            c = re.sub(r',\s*}', '}', s)
            c = re.sub(r',\s*\]', ']', c)
            return json.loads(c, strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Strategy 3: use regex to replace all non-printable chars and normalize
        try:
            c = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
            c = c.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            # Remove any trailing comma before } or ]
            c = re.sub(r',\s*([]}])', r'\1', c)
            return json.loads(c, strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Strategy 4: last resort - manual validation by rebuilding string
        try:
            # Remove any invalid control char not inside strings
            result = []
            in_str = False
            for ch in s:
                if ch == '"' and (len(result) == 0 or result[-1] != '\\'):
                    in_str = not in_str
                if not in_str and (ord(ch) < 32 and ch not in '\t\n\r '):
                    continue
                result.append(ch)
            c = ''.join(result)
            # Remove trailing commas
            c = re.sub(r',\s*([]}])', r'\1', c)
            return json.loads(c, strict=False)
        except (json.JSONDecodeError, TypeError):
            pass
        
        return None
