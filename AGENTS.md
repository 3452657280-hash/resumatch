# ResuMatch 开发指南

## 目录结构

```
resumatch/
├── app/           # 全局配置和数据模型
├── backend/       # FastAPI 后端
├── services/      # 核心业务逻辑（解析、RAG、匹配）
├── frontend.py    # Streamlit 单文件前端
├── data/          # 数据持久化（不提交 git）
├── outputs/       # 输出文件（不提交 git）
```

## 运行方式

- 本地开发：`./run.sh all`（同时启动 API 8000 + Streamlit 8501）
- Docker：`docker compose up --build`

## API Key

通义千问 API Key 配置在 `.env` 文件中，不要提交到 git。

## 代码风格

- 使用 Python 类型注解
- 字符串用双引号
- 函数需要 docstring
- 遵循 FastAPI + Pydantic 的依赖注入模式
