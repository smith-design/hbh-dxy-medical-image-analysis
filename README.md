# 皮肤病变多模态医学影像分析系统

基于 Qwen2-VL 多模态大模型的皮肤病变诊断辅助系统

## 项目概述

本项目使用 HAM10000 开源皮肤病变数据集，基于 Qwen2-VL-7B 多模态基座模型进行微调，实现皮肤病变的智能分类和诊断辅助。

### 数据集信息

- **数据集**: HAM10000 (Human Against Machine with 10000 training images)
- **数据量**: 10,015 张皮肤镜图像
- **疾病类型**: 7 种常见皮肤病变
  - `akiec`: 光化性角化病和上皮内癌 (327 张)
  - `bcc`: 基底细胞癌 (514 张)
  - `bkl`: 良性角化病变 (1,099 张)
  - `df`: 皮肤纤维瘤 (115 张)
  - `mel`: 黑色素瘤 (1,113 张)
  - `nv`: 黑色素痣 (6,705 张)
  - `vasc`: 血管病变 (142 张)

### 技术栈

- **基座模型**: Qwen2-VL-7B-Instruct (通义千问视觉语言模型)
- **微调框架**: LLaMA-Factory (支持 LoRA/QLoRA 高效微调)
- **Web 框架**: Gradio (交互式推理界面)
- **深度学习**: PyTorch, Transformers

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── requirements.txt          # Python 依赖
├── datasets/                 # 原始数据集
│   └── archive (6)/
│       ├── HAM10000_images_part_1/
│       ├── HAM10000_images_part_2/
│       └── HAM10000_metadata.csv
├── data/                     # 处理后的数据
│   └── processed/           # 转换为 LLaMA-Factory 格式
├── src/                      # 源代码
│   ├── data_preprocessing.py    # 数据预处理
│   └── inference.py             # 模型推理
├── configs/                  # 配置文件
│   └── qwen2vl_lora.yaml    # LoRA 微调配置
├── scripts/                  # 脚本
│   ├── train.sh             # 训练脚本
│   └── prepare_data.sh      # 数据准备脚本
├── models/                   # 模型保存目录
│   └── checkpoints/         # 训练检查点
└── web/                      # Web 界面
    └── app.py               # Gradio 应用

```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n skin-lesion python=3.10
conda activate skin-lesion

# 安装依赖
pip install -r requirements.txt

# 克隆 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
cd ..
```

### 2. 数据预处理

```bash
# 将 HAM10000 数据转换为 LLaMA-Factory 格式
python src/data_preprocessing.py
```

### 3. 模型微调

```bash
# 使用 LoRA 微调 Qwen2-VL
bash scripts/train.sh
```

### 4. 启动 Web 界面

```bash
# 启动 Gradio 应用
python web/app.py
```

## 模型微调说明

### 微调方法

- **LoRA (Low-Rank Adaptation)**: 高效参数微调，只训练少量参数
- **训练参数**:
  - Batch size: 4
  - Learning rate: 5e-5
  - Epochs: 3-5
  - LoRA rank: 8
  - LoRA alpha: 16

### 训练任务

模型将学习：
1. 识别皮肤病变类型
2. 描述病变特征（颜色、形状、边界等）
3. 提供初步诊断建议
4. 回答相关医学问题

## Web 界面功能

- 上传皮肤病变图像
- 实时推理和分类
- 显示诊断结果和置信度
- 提供病变特征描述
- 支持多轮对话问答

## 注意事项

⚠️ **免责声明**: 本系统仅用于研究和教育目的，不能替代专业医生的诊断。任何医疗决策应咨询专业医疗人员。

## 参考资源

- [HAM10000 数据集](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [Qwen2-VL 模型](https://github.com/QwenLM/Qwen2-VL)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Gradio 文档](https://www.gradio.app/)

## License

本项目遵循 MIT License。数据集使用请遵循 HAM10000 原始许可。
