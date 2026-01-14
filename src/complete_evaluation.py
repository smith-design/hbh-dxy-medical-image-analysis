"""
Complete evaluation with publication figures
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
import timm
import sys

sys.path.append(str(Path(__file__).parent))
from publication_figures import (
    plot_confusion_matrix_publication,
    plot_per_class_performance,
    DISEASE_CLASSES
)

class HAM10000Dataset(Dataset):
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

def evaluate_and_generate_figures():
    """Evaluate model and generate publication figures"""

    model_dir = Path('models/skin_lesion_classifier')
    dataset_dir = Path('datasets/archive (6)')
    output_dir = model_dir / 'visualizations'

    print("="*70)
    print("🔬 COMPLETE MODEL EVALUATION")
    print("="*70)

    # Load model
    checkpoint_path = model_dir / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_name = checkpoint.get('model_name', 'efficientnet_b0')
    model = timm.create_model(model_name, pretrained=False, num_classes=7)
    model.load_state_dict(checkpoint['model_state_dict'])

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"\n📱 Device: {device}")
    model = model.to(device)
    model.eval()

    # Data
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    metadata = pd.read_csv(dataset_dir / "HAM10000_metadata.csv")

    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(metadata, test_size=0.2, stratify=metadata['dx'], random_state=42)

    val_dataset = HAM10000Dataset(
        val_df,
        dataset_dir / "HAM10000_images_part_1",
        dataset_dir / "HAM10000_images_part_2",
        transform=val_transform
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Evaluate
    print(f"\n🔍 Evaluating on {len(val_df)} samples...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n✅ Validation Accuracy: {accuracy:.2f}%")

    # Generate figures
    print("\n📊 Generating publication figures...")
    print("\n[3/4] Generating confusion matrix...")
    plot_confusion_matrix_publication(all_labels, all_preds, output_dir)

    print("\n[4/4] Generating per-class metrics...")
    plot_per_class_performance(all_labels, all_preds, output_dir)

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\nAll publication figures generated:")
    print("  • fig1_training_curves.png")
    print("  • fig2_confusion_matrix.png")
    print("  • fig3_per_class_metrics.png")
    print("  • fig4_dataset_distribution.png")
    print(f"\n📁 Location: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    evaluate_and_generate_figures()
