"""API 路由定义。"""
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.config import settings
from app.models import ErrorResponse, MatchReport, MatchRequest, UploadResponse
from services import ResumeMatcher, ResumeParser, ResumeRAG
from services.history import HistoryService

router = APIRouter()


def _get_rag(request: Request) -> ResumeRAG:
    """依赖注入：获取 ResumeRAG 实例。"""
    embeddings = request.app.state.embeddings
    return ResumeRAG(embeddings, settings.chroma_persist_dir)


def _get_parser() -> ResumeParser:
    return ResumeParser(settings.upload_dir)


def _get_matcher(request: Request) -> ResumeMatcher:
    llm = request.app.state.llm
    rag = _get_rag(request)
    return ResumeMatcher(llm, rag)


@router.post(
    "/resumes/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
)
async def upload_resume(
    file: UploadFile = File(...),
    parser: ResumeParser = Depends(_get_parser),
    rag: ResumeRAG = Depends(_get_rag),
):
    """上传简历文件（PDF / DOCX），解析并向量化存入 ChromaDB。"""
    if file.filename is None:
        raise HTTPException(400, detail="文件名不能为空")

    content = await file.read()
    content_hash = ResumeRAG.compute_content_hash(content)
    if rag.exists_by_content_hash(content_hash):
        raise HTTPException(400, detail="该简历已上传，请勿重复提交")

    try:
        parsed = parser.save_and_parse(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    chunk_count = rag.add_resume(
        parsed["resume_id"], parsed["filename"], parsed["text"],
        content_hash=content_hash,
    )

    return UploadResponse(
        resume_id=parsed["resume_id"],
        filename=parsed["filename"],
        page_count=parsed["page_count"],
        text_preview=parsed["text"][:200],
    )


@router.post(
    "/match",
    response_model=MatchReport,
    responses={400: {"model": ErrorResponse}},
)
async def match_resumes(
    request: MatchRequest,
    matcher: ResumeMatcher = Depends(_get_matcher),
):
    """根据 JD 匹配单份简历。"""
    all_resumes = matcher.rag.get_all_resumes()
    all_ids = [r["resume_id"] for r in all_resumes]
    if request.resume_id not in all_ids:
        raise HTTPException(400, detail=f"简历 ID {request.resume_id} 不存在，请先上传")

    result = matcher.match_single(
        request.resume_id,
        filename="",
        jd_text=request.jd_text,
    )
    report = MatchReport(results=[result], total_resumes=1)
    HistoryService.save(request.jd_text, "single", report)
    return report


@router.post(
    "/match-all",
    response_model=MatchReport,
    responses={400: {"model": ErrorResponse}},
)
async def match_all_resumes(
    request: MatchRequest,
    matcher: ResumeMatcher = Depends(_get_matcher),
):
    """根据 JD 匹配所有简历。"""
    all_resumes = matcher.rag.get_all_resumes()
    all_ids = [r["resume_id"] for r in all_resumes]
    if not all_ids:
        raise HTTPException(400, detail="向量库中没有简历，请先上传")

    results = []
    for rid in all_ids:
        result = matcher.match_single(rid, filename="", jd_text=request.jd_text)
        results.append(result)

    results.sort(key=lambda r: r.overall_score, reverse=True)
    report = MatchReport(results=results, total_resumes=len(results))
    HistoryService.save(request.jd_text, "batch", report)
    return report


@router.get("/resumes", response_model=list[dict])
async def list_resumes(rag: ResumeRAG = Depends(_get_rag)):
    """列出所有已上传的简历。"""
    return rag.get_all_resumes()


@router.delete("/resumes/{resume_id}")
async def delete_resume(
    resume_id: str,
    rag: ResumeRAG = Depends(_get_rag),
):
    """删除指定简历及其向量数据。"""
    success = rag.delete_resume(resume_id)
    if not success:
        raise HTTPException(404, detail=f"简历 {resume_id} 不存在")
    return {"status": "ok", "resume_id": resume_id}


@router.get("/history")
async def list_history(page: int = 1, page_size: int = 20):
    """列出历史分析记录（摘要）。"""
    records = HistoryService.list(page=page, page_size=page_size)
    total = HistoryService.count()
    return {"records": records, "total": total, "page": page, "page_size": page_size}


@router.get("/history/{record_id}")
async def get_history(record_id: str):
    """获取单条历史分析详情报��。"""
    record = HistoryService.get(record_id)
    if record is None:
        raise HTTPException(404, detail="记录不存在")
    return record


@router.delete("/history/{record_id}")
async def delete_history(record_id: str):
    """删除一条历史记录。"""
    success = HistoryService.delete(record_id)
    if not success:
        raise HTTPException(404, detail="记录不存在")
    return {"status": "ok"}
