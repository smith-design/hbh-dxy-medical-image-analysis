"""
Literature-Quality Advanced Visualizations
Includes: ROC Curves, PR Curves, t-SNE, Grad-CAM, Class Activation Maps
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import timm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import cv2
import warnings

warnings.filterwarnings("ignore")

# Publication-quality settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['patch.linewidth'] = 1

# Color schemes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#6A4C93',
    'light': '#E5E5E5'
}

DISEASE_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DISEASE_NAMES_EN = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

class HAM10000Dataset(Dataset):
    """HAM10000 Dataset"""

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

        # Find image
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

        return image, label, img_id

def plot_roc_curves_advanced(y_true, y_probs, output_dir):
    """Plot advanced ROC curves with zoomed-in view"""

    output_dir = Path(output_dir)
    n_classes = len(DISEASE_CLASSES)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot micro and macro averages
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (area = {roc_auc["micro"]:0.3f})',
             color='deeppink', linestyle=':', linewidth=3)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC (area = {roc_auc["macro"]:0.3f})',
             color='navy', linestyle=':', linewidth=3)

    # Plot each class
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{DISEASE_NAMES_EN[DISEASE_CLASSES[i]]} (area = {roc_auc[i]:0.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'fig5_roc_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure 5 saved: {save_path}")
    plt.close()

def plot_pr_curves_advanced(y_true, y_probs, output_dir):
    """Plot advanced Precision-Recall curves"""

    output_dir = Path(output_dir)
    n_classes = len(DISEASE_CLASSES)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])

    # Micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_probs.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin, y_probs, average="micro")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    plt.plot(recall["micro"], precision["micro"], color='gold', lw=3,
             linestyle=':', label=f'Micro-average (AP = {average_precision["micro"]:0.3f})')

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{DISEASE_NAMES_EN[DISEASE_CLASSES[i]]} (AP = {average_precision[i]:0.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title('Precision-Recall Curves', fontweight='bold')
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'fig6_pr_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure 6 saved: {save_path}")
    plt.close()

def plot_tsne_visualization(features, labels, output_dir):
    """Generate t-SNE visualization of learned features"""

    output_dir = Path(output_dir)
    
    print("\n🔄 Running t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(DISEASE_CLASSES)))
    
    for i, class_name in enumerate(DISEASE_CLASSES):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=DISEASE_NAMES_EN[class_name],
                   alpha=0.6, s=20, edgecolors='none')

    plt.title('t-SNE Visualization of Feature Space', fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    save_path = output_dir / 'fig7_tsne_features.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure 7 saved: {save_path}")
    plt.close()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        output[0, class_idx].backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]
        
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(torch.tensor(heatmap)) if torch.max(torch.tensor(heatmap)) != 0 else 1
        
        return heatmap

def generate_gradcam_visualizations(model, dataset, device, output_dir, num_samples=5):
    """Generate Grad-CAM visualizations for sample images"""
    
    output_dir = Path(output_dir)
    gradcam_dir = output_dir / 'gradcam_samples'
    gradcam_dir.mkdir(exist_ok=True)
    
    # Try to find the target layer (last conv layer)
    # This depends on the model architecture
    target_layer = None
    
    # For EfficientNet, usually the last block
    if hasattr(model, 'conv_head'):
        target_layer = model.conv_head
    elif hasattr(model, 'features'): # VGG-like or generic features
        target_layer = model.features[-1]
    elif hasattr(model, 'layer4'): # ResNet
        target_layer = model.layer4[-1]
        
    if target_layer is None:
        # Try to inspect model structure
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module # Last one found will be used
        
    if target_layer is None:
        print("⚠️ Could not find suitable layer for Grad-CAM. Skipping.")
        return

    grad_cam = GradCAM(model, target_layer)
    
    print(f"\n🔍 Generating Grad-CAM samples (Target Layer: {target_layer.__class__.__name__})...")
    
    # Select samples from each class
    fig, axes = plt.subplots(len(DISEASE_CLASSES), 4, figsize=(16, 4*len(DISEASE_CLASSES)))
    fig.suptitle('Grad-CAM Visualizations by Class', fontsize=20, fontweight='bold', y=0.995)
    
    for i, cls_name in enumerate(DISEASE_CLASSES):
        # Find a sample for this class
        indices = np.where(dataset.metadata['dx'] == cls_name)[0]
        if len(indices) == 0: continue
        
        # Pick a random sample
        idx = np.random.choice(indices)
        img, label, img_id = dataset[idx]
        
        # Original Image for display
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        orig_img = inv_normalize(img).permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        # Prepare for model
        input_tensor = img.unsqueeze(0).to(device)
        
        # Get Grad-CAM
        heatmap = grad_cam(input_tensor, class_idx=label)
        
        # Resize heatmap
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        superimposed_img = heatmap * 0.4 + (orig_img * 255) * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # Plot
        # 1. Original
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Original: {DISEASE_NAMES_EN[cls_name]}', fontsize=10)
        axes[i, 0].axis('off')
        
        # 2. Heatmap
        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title('Grad-CAM Heatmap', fontsize=10)
        axes[i, 1].axis('off')
        
        # 3. Overlay
        axes[i, 2].imshow(superimposed_img)
        axes[i, 2].set_title('Overlay', fontsize=10)
        axes[i, 2].axis('off')
        
        # 4. Prediction Confidence
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            
        # Top 3 preds
        top3_idx = np.argsort(probs)[-3:][::-1]
        text = "Top 3 Predictions:\n"
        for idx in top3_idx:
            text += f"{DISEASE_NAMES_EN[DISEASE_CLASSES[idx]]}: {probs[idx]:.1%}\n"
            
        axes[i, 3].text(0.1, 0.5, text, fontsize=12, va='center', family='monospace')
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    save_path = output_dir / 'fig8_gradcam_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure 8 saved: {save_path}")
    plt.close()

def main_advanced_viz(model_dir='models/skin_lesion_classifier', dataset_dir='datasets/archive (6)'):
    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    output_dir = model_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("🚀 GENERATING ADVANCED LITERATURE-LEVEL VISUALIZATIONS")
    print("="*70)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"📱 Using device: {device}")
    
    # Load Model
    checkpoint_path = model_dir / 'best_model.pth'
    if not checkpoint_path.exists():
        print(f"❌ Model not found: {checkpoint_path}")
        return
        
    print(f"📦 Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get('model_name', 'efficientnet_b0')
    
    model = timm.create_model(model_name, pretrained=False, num_classes=7)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Remove head for feature extraction (t-SNE)
    feature_extractor = timm.create_model(model_name, pretrained=False, num_classes=0) # num_classes=0 removes classifier
    # We need to copy weights, but only the backbone
    # This is tricky without strict mapping, so let's just use the full model and a hook for now
    # Easier approach: Use forward_features method if available
    
    # Data Loading
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"📊 Loading dataset...")
    metadata = pd.read_csv(dataset_dir / "HAM10000_metadata.csv")
    
    # Use a subset for faster t-SNE and viz (e.g., 20% validation set or just random 1000 samples)
    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(metadata, test_size=0.2, stratify=metadata['dx'], random_state=42)
    
    # Limit samples for t-SNE if dataset is huge (optional, here 2000 is fine)
    if len(val_df) > 2000:
        val_df_tsne = val_df.sample(2000, random_state=42)
    else:
        val_df_tsne = val_df
        
    val_dataset = HAM10000Dataset(
        val_df,
        dataset_dir / "HAM10000_images_part_1",
        dataset_dir / "HAM10000_images_part_2",
        transform=val_transform
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Inference
    print(f"🔍 Running inference on {len(val_dataset)} samples...")
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    
    # Hook for features
    features_list = []
    def hook_fn(module, input, output):
        features_list.append(output.flatten(start_dim=1).cpu().detach().numpy())
    
    # Register hook to the layer before classifier
    # For EfficientNet, global_pool is usually the one
    if hasattr(model, 'global_pool'):
        handle = model.global_pool.register_forward_hook(hook_fn)
    else:
        # Fallback: assume forward_features works or just use logits (less ideal for t-SNE but ok)
        print("⚠️  Could not attach hook to global_pool. t-SNE might use logits.")
        handle = None
        
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Inference"):
            images = images.to(device)
            features_list = [] # Reset list for this batch if using hook
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # If hook worked
            if handle and len(features_list) > 0:
                all_features.extend(features_list[0])
            else:
                # Fallback to outputs
                all_features.extend(outputs.cpu().numpy())
                
    if handle: handle.remove()
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)
    
    # 1. ROC Curves
    print("\n[1/4] Generating ROC Curves...")
    plot_roc_curves_advanced(all_labels, all_probs, output_dir)
    
    # 2. PR Curves
    print("\n[2/4] Generating Precision-Recall Curves...")
    plot_pr_curves_advanced(all_labels, all_probs, output_dir)
    
    # 3. t-SNE
    print("\n[3/4] Generating t-SNE Visualization...")
    plot_tsne_visualization(all_features, all_labels, output_dir)
    
    # 4. Grad-CAM (using a fresh dataset instance to access raw images easily if needed, but we passed it)
    print("\n[4/4] Generating Grad-CAM Analysis...")
    # We need a model with gradients enabled for Grad-CAM
    model.train() # Enable gradients
    for param in model.parameters():
        param.requires_grad = True
        
    # Use a small subset for Grad-CAM to keep it fast
    gradcam_df = val_df.groupby('dx').apply(lambda x: x.sample(1)).reset_index(drop=True)
    # We actually need the original dataset class to get the image
    # Let's just pass the validation dataset and let the function pick random samples per class
    generate_gradcam_visualizations(model, val_dataset, device, output_dir)
    
    print("\n" + "="*70)
    print("✅ ADVANCED VISUALIZATIONS GENERATED!")
    print("="*70)

if __name__ == "__main__":
    main_advanced_viz()
