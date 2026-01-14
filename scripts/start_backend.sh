#!/bin/bash

# FastAPI 后端启动脚本

echo "🚀 启动皮肤病变诊断 API 服务..."

# 检查是否在正确的目录
if [ ! -f "web/backend/app.py" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未安装"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
pip list | grep -q fastapi || {
    echo "⚠️  缺少依赖，正在安装..."
    pip install -r web/backend/requirements.txt
}

# 启动服务
echo "✅ 启动 FastAPI 服务..."
echo "📍 API 地址: http://localhost:8000"
echo "📍 API 文档: http://localhost:8000/docs"
echo ""

cd web/backend
python app.py
