"""数据模型 —— 请求 / 响应的 Pydantic 模型。"""
from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    """匹配请求：简历 ID + JD 文本。"""
    resume_id: str
    jd_text: str


class SkillGap(BaseModel):
    skill: str
    status: str = Field(description="matched / missing / weak")
    suggestion: str = ""


class MatchResult(BaseModel):
    resume_id: str
    resume_filename: str
    overall_score: int = Field(ge=0, le=100)
    summary: str
    strengths: list[str] = []
    weaknesses: list[str] = []
    skill_gaps: list[SkillGap] = []
    suggestions: list[str] = []


class MatchReport(BaseModel):
    results: list[MatchResult]
    total_resumes: int
    jd_summary: str = ""


class UploadResponse(BaseModel):
    resume_id: str
    filename: str
    page_count: int
    text_preview: str = ""
    quality_warnings: list[str] = []


class ErrorResponse(BaseModel):
    detail: str


class HistoryRecord(BaseModel):
    """历史分析记录。"""
    id: str = ""
    jd_text: str
    match_type: str = "single"  # single / batch
    created_at: str = ""
    report: MatchReport | None = None
