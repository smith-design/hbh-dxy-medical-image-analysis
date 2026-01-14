"""
完整的模型评估脚本
生成混淆矩阵、ROC曲线、详细性能指标等
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import timm

# 导入可视化函数
import sys
sys.path.append(str(Path(__file__).parent))
from visualize_results import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    generate_performance_report,
    DISEASE_CLASSES,
    DISEASE_NAMES
)

class HAM10000Dataset(Dataset):
    """HAM10000 数据集"""

    def __init__(self, metadata_df, img_dir1, img_dir2, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir1 = Path(img_dir1)
        self.img_dir2 = Path(img_dir2)
        self.transform = transform
        self.label_map = {cls: idx for idx, cls in enumerate(DISEASE_CLASSES)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row['image_id']
        label = self.label_map[row['dx']]

        # 查找图像
        img_path1 = self.img_dir1 / f"{img_id}.jpg"
        img_path2 = self.img_dir2 / f"{img_id}.jpg"

        if img_path1.exists():
            image = Image.open(img_path1).convert('RGB')
        elif img_path2.exists():
            image = Image.open(img_path2).convert('RGB')
        else:
            image = Image.new('RGB', (224, 224), color='gray')

        if self.transform:
            image = self.transform(image)

        return image, label

def evaluate_model(model_dir='models/skin_lesion_classifier', dataset_dir='datasets/archive (6)'):
    """完整评估模型"""

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)

    print("="*60)
    print("🔍 开始完整模型评估")
    print("="*60)

    # 加载模型
    checkpoint_path = model_dir / 'best_model.pth'
    if not checkpoint_path.exists():
        print(f"❌ 未找到模型文件: {checkpoint_path}")
        return

    print(f"\n📦 加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 创建模型
    model_name = checkpoint.get('model_name', 'efficientnet_b0')
    model = timm.create_model(model_name, pretrained=False, num_classes=7)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"📱 使用设备: {device}")
    model = model.to(device)
    model.eval()

    # 数据转换
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据
    print(f"\n📊 加载验证数据...")
    metadata = pd.read_csv(dataset_dir / "HAM10000_metadata.csv")

    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(
        metadata,
        test_size=0.2,
        stratify=metadata['dx'],
        random_state=42
    )

    print(f"✅ 验证集: {len(val_df)} 样本")

    val_dataset = HAM10000Dataset(
        val_df,
        dataset_dir / "HAM10000_images_part_1",
        dataset_dir / "HAM10000_images_part_2",
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # 评估
    print(f"\n🔍 开始评估...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算准确率
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n✅ 验证准确率: {accuracy:.2f}%")

    # 创建输出目录
    output_dir = model_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    # 生成可视化
    print(f"\n📈 生成混淆矩阵...")
    cm = plot_confusion_matrix(all_labels, all_preds, output_dir)

    print(f"\n📊 生成各类别性能指标...")
    report = plot_per_class_metrics(all_labels, all_preds, output_dir)

    # 加载训练历史
    history_path = model_dir / 'training_history.json'
    with open(history_path, 'r') as f:
        history = json.load(f)

    print(f"\n📝 生成性能报告...")
    generate_performance_report(history, report, cm, output_dir)

    print("\n" + "="*60)
    print("✅ 评估完成！")
    print(f"📁 输出目录: {output_dir}")
    print("\n生成的文件:")
    print("  1. confusion_matrix.png - 混淆矩阵")
    print("  2. per_class_metrics.png - 各类别性能指标")
    print("  3. performance_report.md - 详细性能报告")
    print("="*60)

    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'report': report
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='评估模型性能')
    parser.add_argument('--model-dir', type=str, default='models/skin_lesion_classifier',
                        help='模型目录')
    parser.add_argument('--dataset-dir', type=str, default='datasets/archive (6)',
                        help='数据集目录')

    args = parser.parse_args()

    results = evaluate_model(args.model_dir, args.dataset_dir)
