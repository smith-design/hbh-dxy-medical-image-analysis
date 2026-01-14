#!/bin/bash

# HAM10000 皮肤病变数据预处理脚本

echo "🚀 开始数据预处理..."

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未安装，请先安装 Python 3.10+"
    exit 1
fi

# 运行数据预处理
python src/data_preprocessing.py

if [ $? -eq 0 ]; then
    echo "✅ 数据预处理完成！"
    echo "📁 处理后的数据位于: data/processed/"
else
    echo "❌ 数据预处理失败，请检查错误信息"
    exit 1
fi
