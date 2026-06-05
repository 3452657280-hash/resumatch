# ResuMatch 开发指南

## 目录结构

```
resumatch/
├── app/           # 全局配置和数据模型
├── backend/       # FastAPI 后端
├── services/      # 核心业务逻辑（Agent 匹配、RAG、解析、历史）
├── frontend/      # 前端单页面（index.html）
├── data/          # 数据持久化（不提交 git）
```

## 运行方式

- 本地开发：`./run.sh api`（启动 API + 前端，端口 8001）
- Docker：`docker compose up --build`

## API Key

通义千问 API Key 配置在 `.env` 文件中，不要提交到 git。

## 架构说明

- **Agent 匹配器**（`services/matcher.py`）：两阶段式——先提取 JD 关键词，再并发搜索简历片段，最后 LLM 评分 + 自我检查。全程只调 2 次 LLM。
- **RAG 引擎**（`services/rag.py`）：原生 ChromaDB 客户端，无 LangChain 依赖。中文友好文本分块器替代 LangChain 的 RecursiveCharacterTextSplitter。
- **LLM 接入**（`app/config.py`）：纯 httpx 调用 DashScope 的 OpenAI 兼容接口，不依赖 LangChain。

## 代码风格

- 使用 Python 类型注解
- 字符串用双引号
- 函数需要 docstring
- 遵循 FastAPI + Pydantic 的依赖注入模式
