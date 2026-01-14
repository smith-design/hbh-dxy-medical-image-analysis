"""
基于医学预训练模型的皮肤病变分类器
使用迁移学习：医学预训练模型 → HAM10000 微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import timm  # PyTorch Image Models - 包含各种预训练模型

# 疾病类别
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
            # 如果找不到，返回第一个有效图像（避免训练中断）
            print(f"Warning: Image {img_id} not found, using placeholder")
            image = Image.new('RGB', (224, 224), color='gray')

        if self.transform:
            image = self.transform(image)

        return image, label

def create_model(model_name='efficientnet_b0', num_classes=7, pretrained=True):
    """
    创建模型

    Args:
        model_name: 模型名称
            - 'efficientnet_b0': EfficientNet-B0 (推荐)
            - 'resnet18': ResNet18
            - 'resnet34': ResNet34 (更大但更准确)
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
    """

    print(f"📦 创建模型: {model_name}")

  # 使用 timm 加载预训练模型
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )

    print(f"✅ 模型创建完成")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model

def train_model(
    model_name='efficientnet_b0',
    epochs=10,
    batch_size=32,
    lr=0.001,
    device=None
):
    """训练模型"""

    print("="*60)
    print("🚀 开始训练皮肤病变分类器")
    print("="*60)
    print(f"📋 配置:")
    print(f"   模型: {model_name}")
    print(f"   训练轮数: {epochs}")
    print(f"   批次大小: {batch_size}")
    print(f"   学习率: {lr}")
    print("="*60)

    # 设备 - 支持 Apple Silicon
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"\n📱 使用设备: Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"\n📱 使用设备: NVIDIA GPU (CUDA)")
        else:
            device = torch.device('cpu')
            print(f"\n📱 使用设备: CPU")
    else:
        print(f"\n📱 使用设备: {device}")

    if device.type == 'cpu':
        print("⚠️  使用 CPU 训练，速度较慢，请耐心等待...")

    # 数据转换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据
    print("\n📊 加载数据...")
    base_dir = Path("datasets/archive (6)")
    metadata = pd.read_csv(base_dir / "HAM10000_metadata.csv")

    print(f"✅ 总样本数: {len(metadata)}")
    print(f"✅ 类别分布:")
    for cls in DISEASE_CLASSES:
        count = len(metadata[metadata['dx'] == cls])
        print(f"   {DISEASE_NAMES[cls]}: {count}")

    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        metadata,
        test_size=0.2,
        stratify=metadata['dx'],
        random_state=42
    )

    print(f"\n✅ 训练集: {len(train_df)} 样本")
    print(f"✅ 验证集: {len(val_df)} 样本")

    # 创建数据集
    train_dataset = HAM10000Dataset(
        train_df,
        base_dir / "HAM10000_images_part_1",
        base_dir / "HAM10000_images_part_2",
        transform=train_transform
    )

    val_dataset = HAM10000Dataset(
        val_df,
        base_dir / "HAM10000_images_part_1",
        base_dir / "HAM10000_images_part_2",
        transform=val_transform
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # CPU 模式下设为 0
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 创建模型
    print("\n📦 创建模型...")
    model = create_model(model_name=model_name, num_classes=7, pretrained=True)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    save_dir = Path("models/skin_lesion_classifier")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    print("\n" + "="*60)
    print("🔥 开始训练")
    print("="*60)

    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        print("-" * 60)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc="Training", ncols=100)
        for batch_idx, (images, labels) in enumerate(train_bar):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # 更新进度条
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation", ncols=100)
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_bar.set_postfix({
              'loss': f'{loss.item():.4f}',
           'acc': f'{100.*val_correct/val_total:.2f}%'
                })

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # 更新学习率
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"\n📊 Epoch {epoch+1} 结果:")
        print(f"   训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"   验证 - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"   学习率: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'history': history,
                'class_names': DISEASE_CLASSES,
                'model_name': model_name
            }, save_dir / "best_model.pth")

            print(f"   ✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")

    print("\n" + "="*60)
    print("🎉 训练完成！")
    print("="*60)
    print(f"📊 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"💾 模型保存在: {save_dir / 'best_model.pth'}")
    print("="*60)

    # 保存训练历史
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"📈 训练历史保存在: {history_path}")

    return model, history, best_val_acc

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='训练皮肤病变分类器')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet18', 'resnet34'],
                        help='模型名称')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')

    args = parser.parse_args()

    print("\n🎯 开始训练...")
    print(f"   模型: {args.model}")
    print(f"   轮数: {args.epochs}")
    print(f"   批次: {args.batch_size}")
    print(f"   学习率: {args.lr}\n")

    model, history, best_acc = train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print(f"\n✅ 训练完成！最佳准确率: {best_acc:.2f}%")
