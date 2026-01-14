"""
完整的推理管道
整合 BiomedCLIP 分类和魔塔 API 报告生成
"""

import torch
from PIL import Image
from pathlib import Path
import sys
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from biomedclip_classifier import SkinLesionClassifier, DISEASE_CLASSES, DISEASE_NAMES
from modelscope_api import ModelScopeAPI

class SkinLesionDiagnosisSystem:
    """皮肤病变诊断系统"""

    def __init__(self, model_path=None, api_key=None):
        """
        初始化诊断系统

        Args:
            model_path: 分类模型路径
            api_key: 魔塔 API 密钥
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 使用设备: {self.device}")

      # 加载分类模型
        self.classifier = SkinLesionClassifier()
        self.classifier.load_feature_extractor()
        self.classifier = self.classifier.to(self.device)

        if model_path and Path(model_path).exists():
            print(f"📦 加载训练好的模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("⚠️  未找到训练好的模型，使用未训练的分类器")

        self.classifier.eval()

        # 初始化 API 客户端
        self.api_client = ModelScopeAPI(api_key=api_key)

    def classify_image(self, image_path):
        """
        分类图像

        Args:
            image_path: 图像路径

        Returns:
            dict: 包含分类结果的字典
      """
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 预测
        with torch.no_grad():
            logits = self.classifier([image])
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)

        # 获取结果
        disease_code = DISEASE_CLASSES[predicted.item()]
        disease_name = DISEASE_NAMES[disease_code]
        confidence_score = confidence.item()

        # 获取所有类别的概率
        all_probs = {}
        for idx, prob in enumerate(probs[0].tolist()):
            code = DISEASE_CLASSES[idx]
            all_probs[DISEASE_NAMES[code]] = prob

        result = {
            'disease_code': disease_code,
            'disease_name': disease_name,
            'confidence': confidence_score,
            'all_probabilities': all_probs
        }

        return result

    def generate_report(self, classification_result, patient_info=None):
        """
        生成诊断报告

        Args:
            classification_result: 分类结果
            patient_info: 患者信息

        Returns:
            str: 诊断报告
        """
        report = self.api_client.generate_diagnosis_report(
            disease_type=classification_result['disease_code'],
            disease_name=classification_result['disease_name'],
            confidence=classification_result['confidence'],
            patient_info=patient_info
        )

        return report

    def diagnose(self, image_path, patient_info=None):
        """
        完整诊断流程

        Args:
            image_path: 图像路径
            patient_info: 患者信息

        Returns:
            dict: 包含分类结果和报告的字典
        """
        print(f"\n{'='*60}")
        print("🔍 开始诊断...")
        print(f"{'='*60}\n")

        # 步骤1: 图像分类
        print("📊 步骤 1/2: 图像分类...")
        classification_result = self.classify_image(image_path)

        print(f"✅ 分类完成")
        print(f"   病变类型: {classification_result['disease_name']}")
        print(f"   置信度: {classification_result['confidence']:.2%}\n")

        # 显示所有类别概率
        print("📈 各类别概率:")
        sorted_probs = sorted(
            classification_result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for name, prob in sorted_probs[:3]:  # 显示前3个
            print(f"   {name}: {prob:.2%}")
        print()

        # 步骤2: 生成报告
        print("📝 步骤 2/2: 生成诊断报告...")
        report = self.generate_report(classification_result, patient_info)
        print("✅ 报告生成完成\n")

        result = {
            'classification': classification_result,
            'report': report,
            'patient_info': patient_info
        }

        return result

    def save_result(self, result, output_path):
        """保存诊断结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'classification': result['classification'],
                'patient_info': result['patient_info']
            }, f, ensure_ascii=False, indent=2)

        # 保存报告
        report_path = output_path.with_suffix('.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(result['report'])

        print(f"💾 结果已保存:")
        print(f"   JSON: {json_path}")
        print(f"   报告: {report_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='皮肤病变诊断系统')
    parser.add_argument('image', help='图像路径')
    parser.add_argument('--model', help='模型路径', default=None)
    parser.add_argument('--api-key', help='魔塔 API Key', default=None)
    parser.add_argument('--age', type=int, help='患者年龄', default=None)
    parser.add_argument('--sex', help='患者性别', default=None)
    parser.add_argument('--location', help='病变位置', default=None)
    parser.add_argument('--output', help='输出路径', default='output/diagnosis')

    args = parser.parse_args()

    # 检查图像文件
    if not Path(args.image).exists():
        print(f"❌ 图像文件不存在: {args.image}")
        sys.exit(1)

    # 构建患者信息
    patient_info = {}
    if args.age:
        patient_info['age'] = args.age
    if args.sex:
        patient_info['sex'] = args.sex
    if args.location:
        patient_info['localization'] = args.location

    # 创建诊断系统
    system = SkinLesionDiagnosisSystem(
        model_path=args.model,
        api_key=args.api_key
    )

    # 执行诊断
    result = system.diagnose(args.image, patient_info)

    # 显示报告
    print(f"\n{'='*60}")
    print("📋 诊断报告")
    print(f"{'='*60}\n")
    print(result['report'])
    print(f"\n{'='*60}\n")

    # 保存结果
    system.save_result(result, args.output)


if __name__ == "__main__":
    main()
