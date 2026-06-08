# ResuMatch 开发指南

## 目录结构

```
ResuMatch/                 # 项目根目录
├── app/                   # 全局配置和数据模型
│   ├── config.py          # LLM/Embedding 客户端、环境配置
│   └── models.py          # Pydantic 数据模型
├── backend/               # FastAPI 后端
│   ├── main.py            # FastAPI 入口 + 全局异常处理
│   └── routes.py          # API 路由
├── services/              # 核心业务逻辑
│   ├── matcher.py         # Agent 匹配器（4 阶段：提取→搜索→Rerank→打分+自检）
│   ├── rag.py             # ChromaDB 向量检索引擎（含 Rerank）
│   ├── parser.py          # 简历解析 (PDF/DOCX) + LLM 质量验证
│   └── history.py         # SQLite 历史记录
├── frontend/
│   └── index.html         # 前端单页面（HTML + CSS + JS）
├── data/                  # 数据持久化（不上传 git）
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                   # API Key 配置（不上传 git）
```

## 运行方式

- 本地开发：`./run.sh api`（启动 API + 前端，端口 8001）
- Docker：`docker compose up --build`

## API Key

通义千问 API Key 配置在 `.env` 文件中，不要提交到 git。

## 架构说明

### Agent 匹配流程（services/matcher.py）

4 阶段设计：

1. **JD 关键词提取** — LLM 提取结构化标签（硬性条件/软性条件/业务领域/段落建议）
2. **分类并发搜索** — 遍历每个标签，按段落建议优先搜，不够搜全文补齐
3. **Rerank 重排** — RAG 模块用 LLM 对搜索结果重排序，保留 Top-5
4. **评分 + 自检** — LLM 按权重综合打分 → 独立调 LLM 二次审查(self_check)，发现问题调分

全程调 3 次 LLM（提取 1 次 → 打分 1 次 → 自检 1 次），Rerank 由 RAG 模块独立完成。纯 httpx 实现。

### RAG 引擎（services/rag.py）

- 原生 ChromaDB 客户端，无 LangChain 依赖
- 中文友好文本分块器（chunk_size=500, overlap=100）
- 元数据包含 resume_id、filename、chunk_index、section、skills
- 内置 LLM Rerank 重排序方法

### LLM 接入（app/config.py）

- 纯 httpx 调用 DashScope 的 OpenAI 兼容接口
- 不依赖 LangChain 或其他第三方 Agent 框架
- 带自动重试和友好错误提示

## 代码风格

- 使用 Python 类型注解
- 字符串用双引号
- 函数需要 docstring
- 遵循 FastAPI + Pydantic 的依赖注入模式
