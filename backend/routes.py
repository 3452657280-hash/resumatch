"""API 路由定义。"""
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.config import settings
from app.models import ErrorResponse, MatchReport, MatchRequest, UploadResponse
from services import AgentMatcher, ResumeParser, ResumeRAG
from services.history import HistoryService

router = APIRouter()


def _get_rag(request: Request) -> ResumeRAG:
    embedder = request.app.state.embedder
    return ResumeRAG(embedder, settings.chroma_persist_dir)


def _get_parser(request: Request) -> ResumeParser:
    llm = getattr(request.app.state, "llm", None)
    return ResumeParser(settings.upload_dir, llm=llm)


def _get_matcher(request: Request) -> AgentMatcher:
    llm = request.app.state.llm
    rag = _get_rag(request)
    return AgentMatcher(llm, rag)


@router.post(
    "/resumes/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
)
async def upload_resume(
    file: UploadFile = File(...),
    rag: ResumeRAG = Depends(_get_rag),
    parser: ResumeParser = Depends(_get_parser),
):
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

    response = UploadResponse(
        resume_id=parsed["resume_id"],
        filename=parsed["filename"],
        page_count=parsed["page_count"],
        text_preview=parsed["text"][:200],
    )
    # Attach quality warnings as a custom field for the frontend to display
    warnings = parsed.get("quality_warnings", [])
    if warnings:
        setattr(response, "quality_warnings", warnings)
    return response


@router.post(
    "/match",
    response_model=MatchReport,
    responses={400: {"model": ErrorResponse}},
)
async def match_resumes(
    request: MatchRequest,
    matcher: AgentMatcher = Depends(_get_matcher),
):
    all_resumes = matcher.rag.get_all_resumes()
    all_ids = [r["resume_id"] for r in all_resumes]
    if request.resume_id not in all_ids:
        raise HTTPException(400, detail=f"简历 ID {request.resume_id} 不存在，请先上传")

    result = matcher.match_single(request.resume_id, filename="", jd_text=request.jd_text)
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
    matcher: AgentMatcher = Depends(_get_matcher),
):
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
    return rag.get_all_resumes()


@router.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: str, rag: ResumeRAG = Depends(_get_rag)):
    success = rag.delete_resume(resume_id)
    if not success:
        raise HTTPException(404, detail=f"简历 {resume_id} 不存在")
    return {"status": "ok", "resume_id": resume_id}


@router.get("/resumes/{resume_id}/text")
async def get_resume_text(resume_id: str, rag: ResumeRAG = Depends(_get_rag)):
    text = rag.get_resume_text(resume_id)
    if text is None:
        raise HTTPException(404, detail=f"简历 {resume_id} 不存在")
    return {"resume_id": resume_id, "text": text}


@router.get("/history")
async def list_history(page: int = 1, page_size: int = 20):
    records = HistoryService.list(page=page, page_size=page_size)
    total = HistoryService.count()
    return {"records": records, "total": total, "page": page, "page_size": page_size}


@router.get("/history/{record_id}")
async def get_history(record_id: str):
    record = HistoryService.get(record_id)
    if record is None:
        raise HTTPException(404, detail="记录不存在")
    return record


@router.delete("/history/{record_id}")
async def delete_history(record_id: str):
    success = HistoryService.delete(record_id)
    if not success:
        raise HTTPException(404, detail="记录不存在")
    return {"status": "ok"}
