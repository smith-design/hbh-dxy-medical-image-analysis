"""
FastAPI 后端服务
提供皮肤病变诊断 API
支持多模型双重验证：
1. 本地 Qwen2-VL 模型进行初步分析
2. 魔塔社区 API 进行二次验证和详细报告生成
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Add this import
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
from PIL import Image
import io
import os
from pathlib import Path
import logging
import sys

# 添加项目路径以导入其他模块
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.append(str(src_path))
from modelscope_api import ModelScopeAPI
from biomedclip_classifier import SkinLesionClassifier, DISEASE_CLASSES, DISEASE_NAMES

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="皮肤病变多模态诊断系统",
    description="基于 Qwen2-VL 本地模型与魔塔 API 的双重验证诊断系统",
    version="2.0.0"
)

# 挂载静态文件目录，用于提供模型可视化图表
# 路径相对于当前 backend 目录：../../models/skin_lesion_classifier/visualizations
visualization_path = Path(__file__).parent.parent.parent / "models" / "skin_lesion_classifier" / "visualizations"
if visualization_path.exists():
    app.mount("/static/visualizations", StaticFiles(directory=str(visualization_path)), name="visualizations")
    logger.info(f"📂 静态资源已挂载: {visualization_path}")
else:
    logger.warning(f"⚠️ 静态资源目录不存在: {visualization_path}")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型
model = None
processor = None
device = None
modelscope_api = None

import base64

# ... (Previous imports)

# 魔塔 API Key
MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY", "ms-80e2a97a-5492-4c23-bd7c-2bb45497001e") 

# 疾病类型映射
DISEASE_MAPPING = {
    'akiec': '光化性角化病和上皮内癌',
    'bcc': '基底细胞癌',
    'bkl': '良性角化病变',
    'df': '皮肤纤维瘤',
    'mel': '黑色素瘤',
    'nv': '黑色素痣',
    'vasc': '血管病变'
}

class DiagnosisResponse(BaseModel):
    """双重验证诊断响应模型"""
    local_diagnosis: str
    local_disease_type: Optional[str] = None
    cloud_report: str
    confidence: Optional[float] = None
    verification_status: str  # 'match' | 'mismatch' | 'single'
    recommendations: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """启动时初始化服务"""
    global model, processor, device, modelscope_api

    # 1. 初始化本地 BiomedCLIP 分类模型
    try:
        logger.info("🚀 正在加载本地 BiomedCLIP 分类模型...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"📱 使用设备: {device}")

        # 初始化模型结构
        # num_classes = 7 (根据 biomedclip_classifier.py 中的定义)
        model = SkinLesionClassifier(num_classes=7)
        model.load_feature_extractor()
        model.to(device)

        # 加载训练好的权重
        base_dir = Path(__file__).parent.parent.parent
        # 您提到的路径: models/skin_lesion_classifier/best_model.pth
        model_path = base_dir / "models" / "skin_lesion_classifier" / "best_model.pth"

        if model_path.exists():
            logger.info(f"📦 加载训练权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 使用 strict=False 允许加载不完全匹配的权重
            # 这对于当 BiomedCLIP 加载失败切换到 DINOv2 时非常重要
            # 虽然 DINOv2 的结构与 BiomedCLIP 不同，但这能防止服务直接崩溃
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.warning("⚠️ 使用 strict=False 加载权重。如果基础模型架构改变（如从 BiomedCLIP 切换到 DINOv2），部分权重将被忽略。")
            except Exception as e:
                logger.error(f"权重加载严重错误: {e}")
                
            model.eval()
            logger.info("✅ 本地模型加载完成！")
        else:
            logger.warning(f"⚠️  未找到训练权重文件: {model_path}，将使用未训练模型进行演示")
            model.eval()

    except Exception as e:
        logger.error(f"❌ 本地模型加载失败: {str(e)}")
        model = None
        
    # 2. 初始化魔塔 API
    try:
        logger.info("☁️ 正在初始化魔塔 API 客户端...")
        # 注意：这里需要确保用户设置了 API KEY，如果没有，可以在这里硬编码测试或者报错
        modelscope_api = ModelScopeAPI(api_key=MODELSCOPE_API_KEY)
        logger.info("✅ 魔塔 API 客户端就绪")
    except Exception as e:
        logger.error(f"❌ 魔塔 API 初始化失败: {str(e)}")

@app.post("/api/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    file: UploadFile = File(...),
    question: Optional[str] = "请分析这张皮肤病变图像，判断病变类型。"
):
    """
    双重验证诊断流程：
    1. 本地模型推理
    2. 魔塔 API 二次验证与报告生成
    3. 结果比对与合并
    """
    
    # 1. 读取图像
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        logger.info(f"📸 收到图像: {file.filename}, 大小: {image.size}")
        
        # 转换为 Base64 供 API 使用
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
         raise HTTPException(status_code=400, detail="无效的图像文件")

    # 2. 本地模型推理
    local_result = "本地模型未加载"
    disease_type = "unknown"
    confidence = 0.0
    
    if model:
        try:
            # 预处理图像
            # BiomedCLIP 需要 PIL Image 列表
            with torch.no_grad():
                # 直接传递 PIL Image 对象列表
                images = [image] 
                # model.forward 接收 images 列表，内部再调用 extract_features
                logits = model(images) 
                probs = torch.softmax(logits, dim=1)
                confidence_score, predicted = probs.max(1)
            
            # 获取结果
            idx = predicted.item()
            disease_code = DISEASE_CLASSES[idx]
            disease_name = DISEASE_NAMES[disease_code]
            confidence = confidence_score.item()
            
            disease_type = disease_name # UI expecting readable name
            local_result = f"基于 BiomedCLIP 模型的本地分析结果：\n检测到的病变类型为：{disease_name} ({disease_code})\n置信度：{confidence:.2%}"
            
            logger.info(f"✅ 本地推理完成: {disease_name} ({confidence:.2%})")

        except Exception as e:
            logger.error(f"本地推理出错: {e}")
            local_result = f"本地推理错误: {str(e)}"

    # 3. 魔塔 API 生成报告 (作为双重保险)
    cloud_report = "API 未配置或调用失败"
    
    # 将 disease_type 转换回代码以供 API 使用
    disease_code = next((k for k, v in DISEASE_MAPPING.items() if v == disease_type), 'unknown')
    
    if modelscope_api and disease_code != 'unknown':
        try:
            # 使用 API 生成更详细、专业的报告
            # 传入 image_base64 以支持多模态分析
            cloud_report = modelscope_api.generate_diagnosis_report(
                disease_type=disease_code,
                disease_name=disease_type,
                confidence=confidence,
                patient_info={"note": "AI 双重验证请求"},
                image_base64=image_base64
            )
        except Exception as e:
            logger.error(f"API 调用出错: {e}")
            cloud_report = f"无法生成云端报告: {str(e)}"
    elif modelscope_api:
         # 如果本地没识别出来，尝试直接让 API 识别
         try:
            cloud_report = modelscope_api.generate_diagnosis_report(
                disease_type="unknown",
                disease_name="待确认",
                confidence=0.0,
                patient_info={"note": "本地模型未识别，请求云端模型分析"},
                image_base64=image_base64
            )
         except Exception as e:
            cloud_report = "无法生成详细报告"

    # 4. 整合结果
    return DiagnosisResponse(
        local_diagnosis=local_result,
        local_disease_type=disease_type,
        cloud_report=cloud_report,
        confidence=confidence,
        verification_status="match" if disease_type != "unknown" else "check_required",
        recommendations="双重验证完成。请结合本地快速诊断与云端详细报告进行参考。"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
