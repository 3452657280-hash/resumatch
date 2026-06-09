# ResuMatch — AI Agent 简历匹配系统

> 自研 Agent 驱动的简历智能匹配系统，支持单份/批量简历与职位描述（JD）的深度匹配分析。

## 功能

- 📤 **简历上传 & 解析** — 支持 PDF/DOCX，自动分块（500字/块，100字重叠）并向量化存入 ChromaDB
- 🔍 **Agent 单份匹配** — JD 结构化关键词提取 → 段落感知向量检索 → Cross-Encoder 重排 → LLM 综合评分 + 独立自检
- 📊 **批量匹配** — 智能排序所有简历，快速筛选最佳候选人
- 🤖 **AI 小助手** — 侧边栏悬浮智能问答助手
- 📱 **移动端适配** — 侧边栏折叠，手机可用

## 技术栈

| 层 | 技术 |
|---|---|
| 后端框架 | FastAPI (Python 3.11+) + Pydantic |
| AI Agent | 纯 httpx 调用通义千问 DashScope API（无 LangChain） |
| 向量数据库 | ChromaDB（原生客户端） |
| LLM | 通义千问 qwen-max（仅 2 次调用：提取关键词 + 评分自检） |
| Embedding | 通义千问 text-embedding-v3 |
| 简历解析 | PyMuPDF + python-docx，LLM 验证解析质量 |
| 文本分块 | 中文友好分块器（chunk_size=500, overlap=100） |
| Rerank | BAAI/bge-reranker-v2-m3 Cross-Encoder（本地离线，毫秒级） |
| 前端 | 纯 HTML/CSS/JS 单页面（无框架） |
| 历史存储 | SQLite |
| 部署 | Docker / Docker Compose |

## 快速开始

### 前置条件

- Python 3.11+
- 通义千问 API Key（配置在 `.env` 文件的 `DASHSCOPE_API_KEY`）

### 方式一：本地运行

```bash
# 1. 安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置 API Key
# 编辑 .env 文件确认 DASHSCOPE_API_KEY 正确

# 3. 启动 API + 前端
./run.sh api
```

### 方式二：Docker

```bash
docker compose up --build
```

### 访问地址

| 服务 | 地址 |
|---|---|
| 前端页面 | http://localhost:8001 |
| API 文档 | http://localhost:8001/docs |

> ⚠️ `.env` 文件未提交到 git，首次使用需手动创建并填写 `DASHSCOPE_API_KEY=你的key`

## 项目结构

```
ResuMatch/
├── frontend/
│   └── index.html           # 前端单页面（HTML + CSS + JS）
├── backend/
│   ├── main.py              # FastAPI 入口 + 全局异常处理
│   └── routes.py            # API 路由
├── services/
│   ├── matcher.py           # Agent 匹配器（提取→搜索→Rerank→打分+自检）
│   ├── rag.py               # ChromaDB 向量检索引擎
│   ├── reranker.py          # Cross-Encoder 重排服务
│   ├── parser.py            # 简历解析 (PDF/DOCX) + LLM 质量验证
│   └── history.py           # SQLite 历史记录
├── app/
│   ├── config.py            # 配置、LLM/Embedding 客户端（带重试和异常处理）
│   └── models.py            # Pydantic 数据模型
├── data/                    # 数据目录（上传文件 / Chroma 持久化 / SQLite）
├── AGENTS.md                # 开发者指南
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                     # API Key 配置（不上传 git）
```

## API 接口（部分）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/resumes/upload` | 上传简历（PDF/DOCX） |
| GET | `/api/v1/resumes` | 列出所有简历 |
| DELETE | `/api/v1/resumes/{id}` | 删除简历 |
| POST | `/api/v1/match` | 单份 Agent 匹配 |
| POST | `/api/v1/match-all` | 批量 Agent 匹配 |
| GET | `/api/v1/history` | 历史记录列表 |
| GET/DELETE | `/api/v1/history/{id}` | 查看/删除历史详情 |
| GET | `/health` | 健康检查 |

## Agent 匹配流程

```
用户上传简历发送请求到 FastAPI
  → FastAPI 调用解析简历函数解析，将 PDF 内容转为文字
  → 调用 RAG 按 500/100 分块
  → embedding 向量化
  → 存到 ChromaDB
  → 结果返回前端

用户点击匹配上传 JD，发送请求到 FastAPI
  → FastAPI 调用 LLM 将 JD 进行结构化标签提取（硬性条件/软性条件/业务领域/段落建议）
  → 遍历每个标签作为搜索词
  → embedding 转向量
  → 调用 RAG 在 ChromaDB 检索，优先在段落建议内搜，不够再搜全文
  → 按关键词分组，硬性条件不走重排直接保留 top 2，业务领域和软性条件走 Cross-Encoder 重排取 top 1
  → 汇总 ≤15 条送 LLM 综合评分（技术40%/经验30%/项目20%/软技能10%）
  → LLM 自检（独立调 LLM 复核，发现问题调分）
  → FastAPI 将结果返回前端展示
```

## License

MIT

## 仓库镜像

- GitHub: https://github.com/3452657280-hash/resumatch
- Gitee:  https://gitee.com/wangyu-bo/resumatch

---
> 自研 Agent 框架，无第三方 Agent 框架依赖。全程纯 httpx 实现 LLM 调用，ChromaDB 原生客户端实现向量检索。
