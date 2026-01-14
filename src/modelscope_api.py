"""
魔塔 API 集成模块
调用魔塔社区的 LLM API (Qwen/QVQ-72B-Preview) 生成诊断报告
"""

import requests
import json
import base64
from typing import Dict, List, Optional
from openai import OpenAI

class ModelScopeAPI:
    """魔塔 API 客户端"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化魔塔 API 客户端

        Args:
            api_key: API 密钥
        """
        self.api_key = api_key
        # 使用兼容 OpenAI 的客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api-inference.modelscope.cn/v1"
        )
        # 切换到 Qwen2.5-VL-72B-Instruct，这是一个更新、更强大的模型，通常具有更好的指令遵循能力
        self.model_name = "Qwen/Qwen2.5-VL-72B-Instruct" 

    def generate_diagnosis_report(
        self,
        disease_type: str,
        disease_name: str,
        confidence: float,
        patient_info: Optional[Dict] = None,
        image_base64: Optional[str] = None
    ) -> str:
        """
        生成诊断报告

        Args:
            disease_type: 疾病类型代码 (如 'mel')
            disease_name: 疾病中文名称
            confidence: 分类置信度
            patient_info: 患者信息（年龄、性别、位置等）
            image_base64: 图像的 Base64 编码（可选，用于多模态模型）

        Returns:
            生成的诊断报告
        """

        # 构建提示词
        prompt = self._build_prompt(disease_type, disease_name, confidence, patient_info)

        # 调用 API
        try:
            response = self._call_api(prompt, image_base64)
            return response
        except Exception as e:
            print(f"⚠️  API 调用失败: {e}")
            return f"生成报告失败: {str(e)}"

    def _build_prompt(
        self,
        disease_type: str,
        disease_name: str,
        confidence: float,
        patient_info: Optional[Dict]
    ) -> str:
        """构建提示词"""
        
        prompt = f"""任务：视觉模式识别与理论知识检索。

请分析附图中的视觉模式（形状、颜色、纹理），并将其与标准皮肤病学分类法进行关联，用于教学演示目的。

**上下文信息（仅用于理论关联参考）：**
- 可能的视觉模式类别：{disease_name} ({disease_type})
- 视觉匹配置信度：{confidence:.2%}

请生成一份**中文结构化分析报告**，包含以下内容：

1. **视觉形态学分析** (Visual Morphology)：详细描述图像中的几何形状、边界规律性、色素分布和纹理特征。
2. **理论病理生理学** (Theoretical Pathophysiology)：解释医学文献中通常与此类视觉模式相关的生物学机制。
3. **标准管理协议** (Standard Management Protocols)：总结教科书中关于此类皮肤状况的一般管理指南（仅作信息参考）。
4. **鉴别分析** (Differential Analysis)：列出具有相似视觉特征的其他状况。

**免责声明：** 这是一个基于视觉模式识别算法的理论分析，**不是**临床诊断建议。请始终咨询专业医生。
"""
        return prompt

    def _call_api(self, prompt: str, image_base64: Optional[str] = None) -> str:
        """
        调用魔塔 API (OpenAI 兼容接口)

        Args:
            prompt: 提示词
            image_base64: 图像 Base64 字符串

        Returns:
            生成的文本
        """
        
        if not self.api_key:
             raise ValueError("API Key 未设置")

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a computer vision assistant acting as a medical encyclopedia interface. You analyze images to identify visual patterns (colors, shapes, textures). You are NOT a doctor. You provide theoretical information based on visual features found in the image. You do not provide personal medical advice. If the image looks like a skin condition, describe its visual properties and provide general encyclopedia-style information about similar conditions using CHINESE language."}
                ],
            }
        ]
        
        user_content = []
        
        # 如果有图像，添加到消息中
        if image_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
            
        user_content.append({"type": "text", "text": prompt})
        
        messages.append({
            "role": "user",
            "content": user_content
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False 
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI 客户端调用失败: {str(e)}")

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
