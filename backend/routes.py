"""API 路由定义。"""
import json
import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.models import ErrorResponse, MatchReport, MatchRequest, UploadResponse
from services import AgentMatcher, ResumeParser, ResumeRAG
from services.history import HistoryService

router = APIRouter()
logger = logging.getLogger("resumatch")


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
        sections=parsed.get("sections"),
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


@router.post("/match-stream")
async def match_resumes_stream(
    request: MatchRequest,
    matcher: AgentMatcher = Depends(_get_matcher),
):
    """流式匹配，逐阶段推送进度事件。"""
    all_resumes = matcher.rag.get_all_resumes()
    all_ids = [r["resume_id"] for r in all_resumes]
    if request.resume_id not in all_ids:
        raise HTTPException(400, detail=f"简历 ID {request.resume_id} 不存在，请先上传")

    async def event_stream():
        for event in matcher.match_single_stream(request.resume_id, "", request.jd_text):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") == "result":
                # 保存历史记录
                result_data = event["data"]
                from app.models import MatchResult as MR
                result_obj = MR(**result_data)
                report = MatchReport(results=[result_obj], total_resumes=1)
                HistoryService.save(request.jd_text, "single", report)
            elif event.get("type") == "error":
                return

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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


@router.post("/match-all-stream")
async def match_all_resumes_stream(
    request: MatchRequest,
    matcher: AgentMatcher = Depends(_get_matcher),
):
    """流式批量匹配，逐份简历推送进度。"""
    all_resumes = matcher.rag.get_all_resumes()
    all_ids = [r["resume_id"] for r in all_resumes]
    if not all_ids:
        raise HTTPException(400, detail="向量库中没有简历，请先上传")

    async def event_stream():
        results = []
        total = len(all_ids)
        for idx, rid in enumerate(all_ids, 1):
            yield f"data: {json.dumps({'type': 'progress', 'phase': 'resume', 'current': idx, 'total': total, 'label': f'📄 第 {idx}/{total} 份简历'}, ensure_ascii=False)}\n\n"
            for event in matcher.match_single_stream(rid, "", request.jd_text):
                if event.get("type") == "result":
                    results.append(event["data"])
                elif event.get("type") == "error":
                    # 单份简历失败，继续下一份
                    results.append({
                        "resume_id": rid, "resume_filename": "",
                        "overall_score": 0,
                        "summary": f"匹配失败：{event['message']}",
                        "strengths": [], "weaknesses": [],
                        "skill_gaps": [], "suggestions": [],
                    })
        # 全部完成，按分数排序推送结果
        results.sort(key=lambda r: r.get("overall_score", 0), reverse=True)
        yield f"data: {json.dumps({'type': 'result', 'data': results, 'total': total}, ensure_ascii=False)}\n\n"
        from app.models import MatchResult
        objs = [MatchResult(**r) for r in results]
        report = MatchReport(results=objs, total_resumes=total)
        HistoryService.save(request.jd_text, "batch", report)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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


@router.post("/history/batch-delete")
async def batch_delete_history(ids: list[str]):
    count = HistoryService.batch_delete(ids)
    return {"status": "ok", "deleted": count}


@router.post("/chat")
async def chat(request: Request):
    """AI 助手聊天接口。"""
    body = await request.json()
    message = body.get("message", "")
    if not message:
        raise HTTPException(400, detail="消息不能为空")

    llm = request.app.state.llm

    system_prompt = """你是一个叫「小 R」的 AI 助手，是 ResuMatch 简历匹配系统的内置助手。你说话简洁友好，用中文回答。

你可以回答以下方面的问题：
1. 如何使用系统（上传简历、匹配分析、批量匹配、历史记录）
2. 匹配流程和评分规则（两阶段搜索、技能匹配度40%/经验相关30%/项目质量20%/软技能10%）
3. 技术栈（FastAPI/ChromaDB/通义千问/自研Agent）
4. 查看和分析匹配结果
5. 技术栈和架构

如果问到超出范围的问题，礼貌说不知道，引导用户回到系统相关话题。回答不要太长，2-3句话就好。"""

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ])
        return {"reply": response.strip()}
    except RuntimeError as e:
        raise HTTPException(503, detail=f"AI 服务暂时不可用：{e}")
