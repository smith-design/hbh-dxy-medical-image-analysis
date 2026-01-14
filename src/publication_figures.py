"""
Publication-Quality Visualizations (English Only)
Generate high-quality, publication-ready figures for academic papers
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import json

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

def create_publication_figure():
    """Create a publication-quality figure with proper styling"""
    fig = plt.figure(figsize=(12, 8), dpi=300)
    return fig

def plot_training_curves_publication(history, output_dir):
    """Publication-quality training curves"""

    output_dir = Path(output_dir)
    epochs = np.array(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle('Model Training Performance', fontsize=18, fontweight='bold', y=0.995)

    # 1. Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'o-', color=COLORS['primary'],
             label='Training Loss', linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
    ax1.plot(epochs, history['val_loss'], 's-', color=COLORS['danger'],
             label='Validation Loss', linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('(A) Training and Validation Loss', fontweight='bold', loc='left')
    ax1.legend(frameon=True, shadow=True, fancybox=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. Training and Validation Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'o-', color=COLORS['primary'],
             label='Training Accuracy', linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
    ax2.plot(epochs, history['val_acc'], 's-', color=COLORS['success'],
             label='Validation Accuracy', linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('(B) Training and Validation Accuracy', fontweight='bold', loc='left')
    ax2.legend(frameon=True, shadow=True, fancybox=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. Learning Rate Schedule
    ax3 = axes[1, 0]
    if 'lr' in history and history['lr']:
        ax3.plot(epochs, history['lr'], 'd-', color=COLORS['warning'],
                linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
        ax3.set_xlabel('Epoch', fontweight='bold')
        ax3.set_ylabel('Learning Rate', fontweight='bold')
        ax3.set_title('(C) Learning Rate Schedule', fontweight='bold', loc='left')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    else:
        ax3.text(0.5, 0.5, 'No Learning Rate Data', ha='center', va='center',
                fontsize=14, transform=ax3.transAxes)
        ax3.axis('off')

    # 4. Overfitting Analysis
    ax4 = axes[1, 1]
    overfit_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    colors = [COLORS['success'] if gap < 5 else COLORS['warning'] if gap < 10 else COLORS['danger']
              for gap in overfit_gap]
    bars = ax4.bar(epochs, overfit_gap, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=5, color=COLORS['warning'], linestyle='--', linewidth=2, label='Moderate (5%)')
    ax4.axhline(y=10, color=COLORS['danger'], linestyle='--', linewidth=2, label='Severe (10%)')
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Overfitting Gap (%)', fontweight='bold')
    ax4.set_title('(D) Overfitting Analysis', fontweight='bold', loc='left')
    ax4.legend(frameon=True, shadow=True, fancybox=True)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout()

    save_path = output_dir / 'fig1_training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure 1 saved: {save_path}")
    plt.close()

def plot_confusion_matrix_publication(y_true, y_pred, output_dir):
    """Publication-quality confusion matrix"""

    output_dir = Path(output_dir)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    fig.suptitle('Confusion Matrix Analysis', fontsize=18, fontweight='bold')

    class_names = [DISEASE_NAMES_EN[cls] for cls in DISEASE_CLASSES]

    # 1. Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    ax1.set_title('(A) Confusion Matrix (Counts)', fontweight='bold', loc='left', pad=20)
    ax1.set_xlabel('Predicted Label', fontweight='bold')
    ax1.set_ylabel('True Label', fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # 2. Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, linewidths=0.5, linecolor='gray',
                vmin=0, vmax=1)
    ax2.set_title('(B) Confusion Matrix (Normalized)', fontweight='bold', loc='left', pad=20)
    ax2.set_xlabel('Predicted Label', fontweight='bold')
    ax2.set_ylabel('True Label', fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    save_path = output_dir / 'fig2_confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure 2 saved: {save_path}")
    plt.close()

    return cm

def plot_per_class_performance(y_true, y_pred, output_dir):
    """Publication-quality per-class performance metrics"""

    output_dir = Path(output_dir)

    # Generate classification report
    class_names = [DISEASE_NAMES_EN[cls] for cls in DISEASE_CLASSES]
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Extract metrics
    precision = [report[name]['precision'] for name in class_names]
    recall = [report[name]['recall'] for name in class_names]
    f1 = [report[name]['f1-score'] for name in class_names]
    support = [report[name]['support'] for name in class_names]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.suptitle('Per-Class Performance Metrics', fontsize=18, fontweight='bold')

    x = np.arange(len(class_names))
    width = 0.25

    # 1. Precision, Recall, F1-Score comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width, precision, width, label='Precision',
                    color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, recall, width, label='Recall',
                    color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax1.bar(x + width, f1, width, label='F1-Score',
                    color=COLORS['secondary'], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Disease Class', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('(A) Precision, Recall, and F1-Score', fontweight='bold', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend(frameon=True, shadow=True, fancybox=True)
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. Sample distribution
    ax2 = axes[0, 1]
    bars = ax2.bar(x, support, color=COLORS['info'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Disease Class', fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontweight='bold')
    ax2.set_title('(B) Test Set Distribution', fontweight='bold', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # 3. F1-Score ranking
    ax3 = axes[1, 0]
    f1_sorted_idx = np.argsort(f1)
    sorted_names = [class_names[i] for i in f1_sorted_idx]
    sorted_f1 = [f1[i] for i in f1_sorted_idx]

    colors_sorted = [COLORS['success'] if score > 0.8 else COLORS['warning'] if score > 0.6 else COLORS['danger']
                     for score in sorted_f1]
    bars = ax3.barh(range(len(sorted_names)), sorted_f1, color=colors_sorted,
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_yticks(range(len(sorted_names)))
    ax3.set_yticklabels(sorted_names)
    ax3.set_xlabel('F1-Score', fontweight='bold')
    ax3.set_title('(C) F1-Score Ranking', fontweight='bold', loc='left')
    ax3.set_xlim([0, 1.1])
    ax3.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_f1)):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}', va='center', fontweight='bold')

    # 4. Performance heatmap
    ax4 = axes[1, 1]
    metrics_matrix = np.array([precision, recall, f1])
    sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4,
                xticklabels=class_names, yticklabels=['Precision', 'Recall', 'F1-Score'],
                cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='gray',
                vmin=0, vmax=1)
    ax4.set_title('(D) Performance Heatmap', fontweight='bold', loc='left', pad=20)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    save_path = output_dir / 'fig3_per_class_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure 3 saved: {save_path}")
    plt.close()

    return report

def plot_dataset_distribution_publication(metadata_path, output_dir):
    """Publication-quality dataset distribution"""

    output_dir = Path(output_dir)
    metadata = pd.read_csv(metadata_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.suptitle('HAM10000 Dataset Distribution Analysis', fontsize=18, fontweight='bold')

    disease_counts = metadata['dx'].value_counts()
    disease_counts = disease_counts.reindex(DISEASE_CLASSES)
    class_names = [DISEASE_NAMES_EN[cls] for cls in DISEASE_CLASSES]

    # 1. Bar chart
    ax1 = axes[0, 0]
    colors = plt.cm.Set3(range(len(DISEASE_CLASSES)))
    bars = ax1.bar(range(len(class_names)), disease_counts.values,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Disease Class', fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontweight='bold')
    ax1.set_title('(A) Sample Distribution by Class', fontweight='bold', loc='left')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # 2. Pie chart
    ax2 = axes[0, 1]
    explode = [0.05 if i == disease_counts.values.argmax() else 0 for i in range(len(DISEASE_CLASSES))]
    wedges, texts, autotexts = ax2.pie(disease_counts.values,
                                        labels=class_names,
                                        autopct='%1.1f%%',
                                        colors=colors,
                                        explode=explode,
                                        startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax2.set_title('(B) Class Distribution (Percentage)', fontweight='bold', loc='left')

    # 3. Horizontal bar (sorted)
    ax3 = axes[1, 0]
    sorted_counts = disease_counts.sort_values(ascending=True)
    sorted_names = [DISEASE_NAMES_EN[cls] for cls in sorted_counts.index]
    y_pos = np.arange(len(sorted_names))

    bars = ax3.barh(y_pos, sorted_counts.values, color=colors,
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(sorted_names)
    ax3.set_xlabel('Number of Samples', fontweight='bold')
    ax3.set_title('(C) Class Imbalance Analysis', fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(width)}', ha='left', va='center', fontweight='bold')

    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    total_samples = len(metadata)
    max_class = disease_counts.idxmax()
    min_class = disease_counts.idxmin()
    imbalance_ratio = disease_counts.max() / disease_counts.min()

    stats_text = f"""Dataset Statistics

Total Samples: {total_samples:,}
Number of Classes: {len(DISEASE_CLASSES)}

Most Frequent Class:
  {DISEASE_NAMES_EN[max_class]}
  Samples: {disease_counts[max_class]}

Least Frequent Class:
  {DISEASE_NAMES_EN[min_class]}
  Samples: {disease_counts[min_class]}

Imbalance Ratio: {imbalance_ratio:.2f}:1
Mean Samples: {total_samples/len(DISEASE_CLASSES):.0f}
Median Samples: {disease_counts.median():.0f}
    """

    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue',
                                         alpha=0.3, edgecolor='black', linewidth=2))

    plt.tight_layout()

    save_path = output_dir / 'fig4_dataset_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure 4 saved: {save_path}")
    plt.close()

def generate_publication_figures(model_dir='models/skin_lesion_classifier',
                                 dataset_dir='datasets/archive (6)'):
    """Generate all publication-quality figures"""

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    output_dir = model_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("📊 GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*70)

    # Load training history
    history_path = model_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)

        print("\n[1/4] Generating training curves...")
        plot_training_curves_publication(history, output_dir)

    # Dataset distribution
    metadata_path = dataset_dir / "HAM10000_metadata.csv"
    if metadata_path.exists():
        print("\n[2/4] Generating dataset distribution...")
        plot_dataset_distribution_publication(metadata_path, output_dir)

    print("\n" + "="*70)
    print("✅ PUBLICATION FIGURES GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated figures:")
    print("  • fig1_training_curves.png - Training performance")
    print("  • fig4_dataset_distribution.png - Dataset analysis")
    print("\nNote: Run evaluate_model.py first to generate:")
    print("  • fig2_confusion_matrix.png")
    print("  • fig3_per_class_metrics.png")
    print("="*70)

if __name__ == "__main__":
    generate_publication_figures()
