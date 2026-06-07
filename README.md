# ResuMatch — AI Agent 简历匹配系统

> 自研 Agent 驱动的简历智能匹配系统，支持单份/批量简历与职位描述（JD）的深度匹配分析。

## 功能

- 📤 **简历上传 & 解析** — 支持 PDF/DOCX，自动分块（500字/块，100字重叠）并向量化存入 ChromaDB
- 🔍 **Agent 单份匹配** — JD 关键词提取 → 向量检索 → Rerank 重排 → LLM 综合评分 + 二次审查(self_check)
- 📊 **批量匹配** — 所有简历统一排名，带排名徽标，快速筛选最佳候选人
- 🤖 **AI 小助手** — 侧边栏悬浮智能问答助手
- 📱 **移动端适配** — 侧边栏折叠，手机可用

## 技术栈

| 层 | 技术 |
|---|---|
| 后端框架 | FastAPI (Python 3.11+) + Pydantic |
| AI Agent | 纯 httpx 调用通义千问 DashScope API（无 LangChain） |
| 向量数据库 | ChromaDB（原生客户端） |
| LLM | 通义千问 qwen-max |
| Embedding | 通义千问 text-embedding-v3 |
| 简历解析 | PyMuPDF + python-docx，LLM 验证解析质量 |
| 文本分块 | 中文友好分块器（chunk_size=500, overlap=100） |
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
resumatch/
├── frontend/
│   └── index.html       # 前端单页面（HTML + CSS + JS）
├── backend/
│   ├── main.py           # FastAPI 入口
│   └── routes.py         # API 路由
├── services/
│   ├── matcher.py        # Agent 匹配器（JD提取 → 检索 → Rerank → 评分 + self_check）
│   ├── rag.py            # ChromaDB 向量检索引擎
│   ├── parser.py         # 简历解析 (PDF/DOCX) + LLM 质量验证
│   └── history.py        # SQLite 历史记录
├── app/
│   ├── config.py         # 配置、LLM/Embedding 客户端
│   └── models.py         # Pydantic 数据模型
├── data/                 # 数据目录（上传文件 / Chroma 持久化 / SQLite）
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml        # 项目配置（Python 3.11+）
└── requirements.txt
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

## License

MIT

## 仓库镜像

- GitHub: https://github.com/3452657280-hash/resumatch
- Gitee:  https://gitee.com/wangyu-bo/resumatch

---
> 自研 Agent 框架：两阶段 JD 关键词提取 → ChromaDB 向量检索 → Rerank 重排 → LLM 加权评分 + 二次审查，全程纯 httpx 实现，无第三方 Agent 框架依赖。
