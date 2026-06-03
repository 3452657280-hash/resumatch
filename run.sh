#!/usr/bin/env bash
set -e

echo "=========================================="
echo "  ResuMatch — AI 简历匹配系统"
echo "=========================================="

case "${1:-all}" in
  api|all)
    echo "▶ 启动 API + 前端 (端口 8001)..."
    echo "   http://localhost:8001"
    echo "   API 文档: http://localhost:8001/docs"
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
    ;;
  docker)
    echo "▶ 启动 Docker Compose..."
    docker compose up --build
    ;;
  *)
    echo "用法: ./run.sh [api|docker]"
    exit 1
    ;;
esac
