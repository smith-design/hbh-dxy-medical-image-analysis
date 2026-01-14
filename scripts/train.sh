#!/bin/bash

# Qwen2-VL 模型微调训练脚本

echo "🚀 开始训练 Qwen2-VL 模型..."

# 检查 LLaMA-Factory 是否安装
if [ ! -d "LLaMA-Factory" ]; then
    echo "📦 克隆 LLaMA-Factory..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e .
    cd ..
fi

# 检查数据是否已处理
if [ ! -f "data/processed/train.json" ]; then
    echo "⚠️  未找到处理后的数据，先运行数据预处理..."
    bash scripts/prepare_data.sh
fi

# 复制数据集配置到 LLaMA-Factory
echo "📋 配置数据集..."
cp data/processed/dataset_info.json LLaMA-Factory/data/dataset_info.json

# 创建软链接到数据目录
if [ ! -L "LLaMA-Factory/data/ham10000_skin_lesion" ]; then
    ln -s "$(pwd)/data/processed" LLaMA-Factory/data/ham10000_skin_lesion
fi

# 开始训练
echo "🔥 开始 LoRA 微调..."
cd LLaMA-Factory

llamafactory-cli train ../configs/qwen2vl_lora.yaml

if [ $? -eq 0 ]; then
    echo "✅ 训练完成！"
    echo "📁 模型保存在: models/qwen2vl_ham10000_lora/"
else
    echo "❌ 训练失败，请检查错误信息"
    exit 1
fi

cd ..
