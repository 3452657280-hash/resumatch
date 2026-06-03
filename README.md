# ResuMatch — AI 简历匹配 & 智能分析系统

> 基于 LangChain + RAG 的智能简历匹配工具，支持单份 / 批量简历与职位描述（JD）的深度匹配分析。

## 功能

- 📤 **简历上传 & 解析** — 支持 PDF / DOCX，自动提取文本并向量化存入 ChromaDB
- 🔍 **单份匹配** — 针对一份简历与 JD 进行深度分析（评分、技能差距、改进建议）
- 📊 **批量匹配** — 所有简历统一排名，快速筛选最佳候选人
- 🤖 **AI 分析报告** — 基于通义千问（Qwen）大模型，输出结构化评分和详细建议

## 技术栈

| 层 | 技术 |
|---|---|
| 后端框架 | FastAPI (Python) |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| LLM | 通义千问 qwen-max |
| Embedding | 通义千问 text-embedding-v3 |
| 简历解析 | PyMuPDF + python-docx |
| 前端 | Streamlit |
| 部署 | Docker / Docker Compose |

## 快速开始

### 方式一：本地运行

```bash
# 1. 安装依赖
pip install poetry
poetry install

# 2. 配置 API Key（已配置在 .env 中）
# 编辑 .env 文件确认 DASHSCOPE_API_KEY 正确

# 3. 同时启动后端 + 前端
./run.sh all
```

### 方式二：Docker

```bash
docker compose up --build
```

### 访问地址

| 服务 | 地址 |
|---|---|
| FastAPI | http://localhost:8000 |
| API 文档 | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## 项目结构

```
resumatch/
├── app/
│   ├── config.py       # 全局配置 & LLM 初始化
│   └── models.py       # Pydantic 数据模型
├── backend/
│   ├── main.py         # FastAPI 入口
│   └── routes.py       # API 路由
├── services/
│   ├── parser.py       # 简历解析 (PDF/DOCX)
│   ├── rag.py          # RAG 引擎 (LangChain + Chroma)
│   └── matcher.py      # 匹配分析 (LLM 评分)
├── frontend.py         # Streamlit 前端
├── data/               # 数据目录（上传文件 / Chroma 持久化）
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/resumes/upload` | 上传简历 |
| GET | `/api/v1/resumes` | 列出所有简历 |
| POST | `/api/v1/match` | 单份匹配 |
| POST | `/api/v1/match-all` | 批量匹配 |
| GET | `/health` | 健康检查 |
