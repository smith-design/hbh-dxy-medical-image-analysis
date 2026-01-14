# 基于双重验证的多模态皮肤病变智能分析系统

## 📖 项目简介 (Project Overview)

本项目是一个专为医学科研与教学设计的**皮肤病变智能辅助分析平台**。系统创新性地采用了**“本地专用小模型 + 云端通用大模型”**的双重验证架构，旨在解决单一模型在专业性与泛化能力之间的平衡问题。

通过结合本地微调的 **BiomedCLIP** 模型（针对皮肤镜图像优化）与云端 **Qwen2.5-VL** 多模态大模型（具备海量医学知识），系统不仅能快速识别病变类别，还能生成包含视觉形态学分析、病理生理学解释及鉴别诊断思路的**详细中文学术报告**。

---

## ✨ 核心功能 (Core Features)

### 1. 双重验证机制 (Dual-Verification System)
- **⚡️ 第一重：本地快速诊断 (Local Inference)**
  - **模型架构**：基于 `BiomedCLIP` (PubMedBERT + ViT) 微调的专用分类器。
  - **特点**：速度快、隐私性好、针对皮肤镜数据（HAM10000）深度优化。
  - **作用**：提供即时的良恶性分类和置信度评分。
  
- **☁️ 第二重：云端深度解析 (Cloud Analysis)**
  - **模型架构**：阿里通义千问 `Qwen2.5-VL-72B-Instruct` 多模态大模型。
  - **特点**：具备强大的视觉理解能力和海量医学百科知识。
  - **作用**：生成结构化的中文学术报告，解释“为什么是这个诊断”，提供鉴别诊断思路。

### 2. 多模态分析报告 (Multimodal Diagnosis Report)
系统生成的报告严格遵循医学 **ABCD 规则**（不对称性、边界、颜色、直径），内容涵盖：
- **视觉形态学分析**：描述病变几何特征、纹理及色素分布。
- **理论病理生理学**：解释背后的生物学机制。
- **风险分层**：基于临床指南的风险评估。
- **鉴别诊断**：列出易混淆的其他皮肤状况。

### 3. 高鲁棒性架构 (Robust Architecture)
- **智能降级策略**：若本地 BiomedCLIP 加载失败（如网络问题），系统自动切换至轻量级 **DINOv2** 模型，确保服务不中断。
- **动态维度适配**：自动处理不同预训练模型的特征维度差异（Pad/Truncate），保证分类头兼容。
- **抗拒绝设计**：通过精心设计的 Prompt Engineering，将任务框架化为“视觉模式识别与教学演示”，有效规避大模型的安全拒绝误判，同时保持输出的专业性。

### 4. 现代化可视化前端 (Modern UI/UX)
- **技术栈**：React + TypeScript + Vite + Tailwind CSS v4。
- **交互体验**：
  - 拖拽式图片上传，带有扫描动画特效。
  - 实时双列布局：左侧显示原图与本地结果，右侧流式渲染云端 Markdown 报告。
  - 交互式图表：展示模型在验证集上的性能指标（混淆矩阵、ROC 曲线等）。

---

## 🛠 技术架构 (Technical Architecture)

### 后端 (Backend)
- **框架**：FastAPI (Python)
- **机器学习库**：PyTorch, Transformers, OpenCLIP, Scikit-learn
- **API 集成**：OpenAI SDK (用于调用 ModelScope 兼容接口)
- **关键文件**：
  - `src/biomedclip_classifier.py`: 本地模型定义与特征提取逻辑。
  - `src/modelscope_api.py`: 云端 API 交互与 Prompt 构建。
  - `web/backend/app.py`: FastAPI 服务入口与双重验证流程控制。

### 前端 (Frontend)
- **框架**：React 18
- **样式**：Tailwind CSS v4
- **组件库**：Lucide React (图标), Framer Motion (动画)
- **可视化**：Recharts (图表展示)
- **关键文件**：
  - `web/frontend/src/App.tsx`: 主应用逻辑。
  - `web/frontend/src/components/DiagnosisReport.tsx`: 报告渲染组件。

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备 (Prerequisites)
确保已安装 Python 3.8+ 和 Node.js 16+。

### 2. 启动后端 (Backend Setup)
```bash
# 进入后端目录
cd web/backend

# 安装依赖
pip install -r requirements.txt
pip install -r ../../requirements.txt

# 启动服务 (默认端口 8000)
python app.py
```
> **注意**：首次启动时会自动下载 BiomedCLIP 模型权重（约 780MB），请耐心等待。如果下载失败，系统会自动切换到 DINOv2。

### 3. 启动前端 (Frontend Setup)
```bash
# 进入前端目录
cd web/frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```
访问浏览器控制台输出的地址（通常是 `http://localhost:5173`）。

---

## 🧪 支持检测的病变类别 (Supported Classes)

系统基于 HAM10000 标准数据集训练，支持以下 7 类病变：

| 代号 | 中文名称 | 临床意义 |
| :--- | :--- | :--- |
| **akiec** | 光化性角化病 / 上皮内癌 | 癌前病变，需密切关注 |
| **bcc** | 基底细胞癌 | 最常见的皮肤癌，转移率低但需治疗 |
| **bkl** | 良性角化病 | 如脂溢性角化病，通常良性 |
| **df** | 皮肤纤维瘤 | 良性结节 |
| **mel** | **黑色素瘤** | **高度恶性，需立即就医** |
| **nv** | 黑色素细胞痣 | 普通的“痦子”，良性 |
| **vasc** | 血管病变 | 如血管瘤 |

---

## ⚠️ 免责声明 (Disclaimer)

1.  **非医疗器械**：本系统仅供**医学科研、教学演示及辅助参考**使用，不属于医疗器械。
2.  **不可替代医生**：系统的分析结果基于概率模型，存在误判可能。**任何医疗决定（诊断、用药、手术）必须由持有执照的专业医生做出。**
3.  **数据隐私**：上传的图片仅用于当次推理，系统不会永久存储您的个人健康数据。

---

## 👨‍💻 开发者信息 (Developers)

本项目由多模态医学影像分析团队开发。
如果您在使用中遇到问题，或有合作意向，请联系开发团队。

---

*Copyright © 2026 Skin Lesion Analysis System. All Rights Reserved.*
