"""FastAPI 应用入口。"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings, get_llm, get_embedder
from backend.routes import router

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = get_llm()
    embedder = get_embedder()
    app.state.llm = llm
    app.state.embedder = embedder
    yield


app = FastAPI(
    title="ResuMatch API",
    description="AI 简历匹配 & 智能分析系统",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常捕获，LLM/Embedding 相关错误返回友好提示。"""
    error_msg = str(exc)
    if "API Key" in error_msg:
        return JSONResponse(
            status_code=401,
            content={"detail": "API Key 配置有误，请在 .env 中设置正确的 DASHSCOPE_API_KEY"},
        )
    if "超时" in error_msg:
        return JSONResponse(
            status_code=503,
            content={"detail": "AI 服务响应超时，请稍后重试"},
        )
    if "网络" in error_msg or "连接" in error_msg:
        return JSONResponse(
            status_code=503,
            content={"detail": "无法连接 AI 服务，请检查网络后重试"},
        )
    if "解析" in error_msg or "格式" in error_msg:
        return JSONResponse(
            status_code=500,
            content={"detail": f"数据处理异常：{error_msg}"},
        )
    raise exc

# Serve the frontend static files (must be last to not shadow other routes)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
