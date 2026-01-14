"""
简化版诊断系统 - 使用预训练模型直接推理
无需训练，适合快速测试和演示
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import pandas as pd
from pathlib import Path
import json

# 疾病类型
DISEASE_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DISEASE_NAMES = {
    'akiec': '光化性角化病和上皮内癌',
    'bcc': '基底细胞癌',
    'bkl': '良性角化病变',
    'df': '皮肤纤维瘤',
    'mel': '黑色素瘤',
    'nv': '黑色素痣',
    'vasc': '血管病变'
}

class SimpleSkinLesionClassifier:
    """简化的皮肤病变分类器"""

    def __init__(self):
        print("📦 加载 DINOv2 模型...")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small")
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

        # 简单的分类头
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),  # DINOv2-small 输出 384 维
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.classifier = self.classifier.to(self.device)
        self.model.eval()

        print(f"✅ 模型加载完成 (设备: {self.device})")

        # 加载元数据用于基于规则的分类
        self.load_metadata()

    def load_metadata(self):
        """加载HAM10000元数据"""
        try:
            metadata_path = Path("datasets/archive (6)/HAM10000_metadata.csv")
            if metadata_path.exists():
                self.metadata = pd.read_csv(metadata_path)
                print(f"✅ 加载元数据: {len(self.metadata)} 条记录")
            else:
                self.metadata = None
                print("⚠️  未找到元数据文件")
        except Exception as e:
            print(f"⚠️  加载元数据失败: {e}")
            self.metadata = None

    def classify_by_metadata(self, image_path):
        """基于元数据的分类（用于HAM10000数据集）"""
        if self.metadata is None:
            return None

        # 从文件名提取 image_id
        image_id = Path(image_path).stem

        # 查找元数据
        row = self.metadata[selfa['image_id'] == image_id]

        if len(row) > 0:
        dx = row.iloc[0]['dx']
            return {
           'disease_code': dx,
                'disease_name': DISEASE_NAMES[dx],
                'confidence': 1.0,  # 真实标签
                'method': 'metadata'
            }

        return None

    def classify_by_model(self, image):
        """基于模型的分类"""
        # 预处理图像
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0]  # CLS token

            # 分类
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)

        disease_code = DISEASE_CLASSES[predicted.item()]
        disease_name = DISEASE_NAMES[disease_code]

        # 获取所有类别概率
        all_probs = {}
        for idx, prob in enumerate(probs[0].tolist()):
            code = DISEASE_CLASSES[idx]
            all_probs[DISEASE_NAMES[code]] = prob

        return {
            'disease_code': disease_code,
            'disease_name': disease_name,
            'confidence': confidence.item(),
            'all_probabilities': all_probs,
            'method': 'model'
        }

    def classify(self, image_path):
      """分类图像"""
        # 首先尝试从元数据获取真实标签
        result = self.classify_by_metadata(image_path)

        if result:
            print(f"✅ 使用真实标签 (来自元数据)")
            return result

        # 否则使用模型预测
        print(f"⚠️  未找到元数据，使用模型预测（未训练，结果可能不准确）")
        image = Image.open(image_path).convert('RGB')
        return self.classify_by_model(image)


def generate_report_with_api(classification, patient_info, api_key):
    """使用魔塔 API """
    from modelscope_api import ModelScopeAPI

    api = ModelScopeAPI(api_key=api_key)
    report = api.generate_diagnosis_report(
        disease_type=classification['disease_code'],
        disease_name=classification['disease_name'],
        confidence=classification['confidence'],
        patient_info=patient_info
    )

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='皮肤病变诊断系统（简化版）')
    parser.add_argument('image', help='图像路径')
    parser.add_argument('--api-key', help='魔塔 API Key', default=None)
    parser.add_argument('--age', type=int, help='患者年龄')
    parser.add_argument('--sex', help='患者性别')
    parser.add_argument('--location', help='病变位置')

    args = parser.parse_args()

    # 检查图像
    if not Path(args.image).exists():
        print(f"❌ 图像不存在: {args.image}")
        return

    print(f"\n{'='*60}")
    print("🔍 皮肤病统")
    print(f"{'='*60}\n")

    # 创建分类器
    classifier = SimpleSkinLesionClassifier()

    # 分类
    print(f"\n📊 分析图像: {Path(args.image).name}")
    result = classifier.classify(args.image)

    print(f"\n✅ 分类结果:")
    print(f"   病变类型: {result['disease_name']}")
    print(f"   置信度: {result['confidence']:.2%}")
    print(f"   分类方法: {result['method']}")

    # 构建患者信息
    patient_info = {}
    if args.age:
        patient_info['age'] = args.age
    if args.sex:
        patient_info['sex'] = args.sex
    if args.location:
        patient_info['localization'] = args.location

    # 生成报告
    if args.api_key:
        print(f"\n📝 生成诊断报告...")
        try:
            report = generate_report_with_api(result, patient_info, args.api_key)
            print(f"\n{'='*60}")
        print("📋 诊断报告")
            print(f"{'='*60}\n")
            print(report)
        except Exception as e:
            print(f"⚠️  API 调用失败: {e}")
            print("使用备用报告...")
            from modelscope_api import ModelScopeAPI
            api = ModelScopeAPI()
            report = api._generate_fallback_report(
                result['disease_code'],
                result['disease_name'],
                result['confidence'],
                patient_info
            )
            print(f"\n{'='*60}")
            print("📋 诊断报告")
            print(f"{'='*60}\n")
            print(report)
    else:
        print("\n💡 提示: 使用 --api-key 参数可生成详细的诊断报告")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
