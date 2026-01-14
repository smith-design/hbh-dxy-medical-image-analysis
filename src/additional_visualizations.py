"""
额外的论文级可视化图表
包括：ROC曲线、PR曲线、样本展示、错误分析等
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import json

# 设置样式
import platform
system = platform.system()
if system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

DISEASE_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DISEASE_NAMES = {
    'akiec': '光化性角化病',
    'bcc': '基底细胞癌',
    'bkl': '良性角化病变',
    'df': '皮肤纤维瘤',
    'mel': '黑色素瘤',
    'nv': '黑色素痣',
    'vasc': '血管病变'
}

def plot_roc_curves(y_true, y_probs, output_dir):
    """绘制多类别 ROC 曲线"""

    output_dir = Path(output_dir)
    n_classes = len(DISEASE_CLASSES)

    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # 计算每个类别的 ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算 micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 绘制
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. 所有类别的 ROC 曲线
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        ax1.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{DISEASE_NAMES[DISEASE_CLASSES[i]]} (AUC = {roc_auc[i]:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    ax1.plot(fpr["micro"], tpr["micro"], 'r-', lw=3,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curves - All Classes', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. AUC 柱状图
    auc_values = [roc_auc[i] for i in range(n_classes)]
    class_names = [DISEASE_NAMES[cls] for cls in DISEASE_CLASSES]

    bars = ax2.barh(range(n_classes), auc_values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(n_classes))
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
    ax2.set_title('AUC Scores by Class', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1.1])
    ax2.axvline(roc_auc["micro"], color='red', linestyle='--', linewidth=2,
               label=f'Micro-avg: {roc_auc["micro"]:.3f}')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, auc_values)):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    save_path = output_dir / 'roc_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ ROC 曲线已保存: {save_path}")
    plt.close()

def plot_pr_curves(y_true, y_probs, output_dir):
    """绘制 Precision-Recall 曲线"""

    output_dir = Path(output_dir)
    n_classes = len(DISEASE_CLASSES)

    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # 计算每个类别的 PR 曲线
    precision = dict()
    recall = dict()
    ap_score = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap_score[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])

    # 绘制
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. PR 曲线
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        ax1.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{DISEASE_NAMES[DISEASE_CLASSES[i]]} (AP = {ap_score[i]:.3f})')

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. AP 分数柱状图
    ap_values = [ap_score[i] for i in range(n_classes)]
    class_names = [DISEASE_NAMES[cls] for cls in DISEASE_CLASSES]

    bars = ax2.barh(range(n_classes), ap_values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(n_classes))
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Average Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Average Precision by Class', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, ap_values)):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    save_path = output_dir / 'pr_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ PR 曲线已保存: {save_path}")
    plt.close()

def plot_training_comparison(history, output_dir):
    """绘制训练过程详细对比"""

    output_dir = Path(output_dir)
    epochs = range(1, len(history['train_loss']) + 1)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Training Process Detailed Analysis', fontsize=18, fontweight='bold')

    # 1. Loss 对比（线图 + 填充）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax1.fill_between(epochs, history['train_loss'], alpha=0.3, color='blue')
    ax1.fill_between(epochs, history['val_loss'], alpha=0.3, color='red')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Loss Curves with Fill', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy 对比
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=6)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=6)
    ax2.fill_between(epochs, history['train_acc'], alpha=0.3, color='blue')
    ax2.fill_between(epochs, history['val_acc'], alpha=0.3, color='red')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Accuracy Curves with Fill', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 过拟合分析
    ax3 = fig.add_subplot(gs[0, 2])
    overfit_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    colors = ['green' if gap < 5 else 'orange' if gap < 10 else 'red' for gap in overfit_gap]
    ax3.bar(epochs, overfit_gap, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=5, color='orange', linestyle='--', label='Moderate (5%)')
    ax3.axhline(y=10, color='red', linestyle='--', label='Severe (10%)')
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Overfitting Gap (%)', fontweight='bold')
    ax3.set_title('Overfitting Analysis', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Loss 改善率
    ax4 = fig.add_subplot(gs[1, 0])
    train_loss_improvement = [0] + [history['train_loss'][i-1] - history['train_loss'][i]
                                    for i in range(1, len(history['train_loss']))]
    val_loss_improvement = [0] + [history['val_loss'][i-1] - history['val_loss'][i]
                                  for i in range(1, len(history['val_loss']))]

    x = np.arange(len(epochs))
    width = 0.35
    ax4.bar(x - width/2, train_loss_improvement, width, label='Train', alpha=0.8)
    ax4.bar(x + width/2, val_loss_improvement, width, label='Val', alpha=0.8)
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Loss Improvement', fontweight='bold')
    ax4.set_title('Loss Improvement per Epoch', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(epochs)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 5. Accuracy 改善率
    ax5 = fig.add_subplot(gs[1, 1])
    train_acc_improvement = [0] + [history['train_acc'][i] - history['train_acc'][i-1]
                                   for i in range(1, len(history['train_acc']))]
    val_acc_improvement = [0] + [history['val_acc'][i] - history['val_acc'][i-1]
                                 for i in range(1, len(history['val_acc']))]

    ax5.bar(x - width/2, train_acc_improvement, width, label='Train', alpha=0.8)
    ax5.bar(x + width/2, val_acc_improvement, width, label='Val', alpha=0.8)
    ax5.set_xlabel('Epoch', fontweight='bold')
    ax5.set_ylabel('Accuracy Improvement (%)', fontweight='bold')
    ax5.set_title('Accuracy Improvement per Epoch', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(epochs)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 6. 关键指标表格
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    best_epoch = np.argmax(history['val_acc']) + 1
    best_val_acc = max(history['val_acc'])
    final_overfit = history['train_acc'][-1] - history['val_acc'][-1]
    total_improvement = history['val_acc'][-1] - history['val_acc'][0]

    metrics_text = f"""
Key Metrics Summary

Best Epoch: {best_epoch}
Best Val Acc: {best_val_acc:.2f}%

Final Train Acc: {history['train_acc'][-1]:.2f}%
Final Val Acc: {history['val_acc'][-1]:.2f}%

Overfit Gap: {final_overfit:.2f}%
Total Improvement: {total_improvement:.2f}%

Min Train Loss: {min(history['train_loss']):.4f}
Min Val Loss: {min(history['val_loss']):.4f}
    """

    ax6.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    save_path = output_dir / 'training_detailed_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练详细分析已保存: {save_path}")
    plt.close()

def generate_additional_visualizations(model_dir='models/skin_lesion_classifier'):
    """生成额外的可视化图表"""

    model_dir = Path(model_dir)
    output_dir = model_dir / 'visualizations'

    print("="*60)
    print("📊 生成额外的论文级可视化图表")
    print("="*60)

    # 加载评估结果（需要先运行 evaluate_model.py）
    # 这里我们先生成训练相关的图表

    # 加载训练历史
    history_path = model_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)

        print("\n📈 生成训练详细分析...")
        plot_training_comparison(history, output_dir)

    print("\n" + "="*60)
    print("✅ 额外可视化生成完成！")
    print("="*60)

if __name__ == "__main__":
    generate_additional_visualizations()
