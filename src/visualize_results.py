"""
训练结果可视化和性能分析
生成训练曲线、混淆矩阵、性能报告等
"""

import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

# 设置中文字体 - 修复中文显示问题
import platform
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
elif system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# 疾病类别
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

def plot_training_curves(history, output_dir):
    """绘制训练曲线"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('训练过程可视化', fontsize=20, fontweight='bold', y=0.995)

    # 1. 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='训练损失', linewidth=2, markersize=6)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='验证损失', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('损失曲线', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 准确率曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='训练准确率', linewidth=2, markersize=6)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='验证准确率', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('准确率曲线', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 学习率变化
    if 'lr' in history and history['lr']:
        axes[1, 0].plot(epochs, history['lr'], 'g-d', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('学习率变化', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, '无学习率数据', ha='center', va='center', fontsize=14)
        axes[1, 0].axis('off')

    # 4. 训练 vs 验证对比
    x = np.arange(len(epochs))
    width = 0.35

    axes[1, 1].bar(x - width/2, history['train_acc'], width, label='训练准确率', alpha=0.8)
    axes[1, 1].bar(x + width/2, history['val_acc'], width, label='验证准确率', alpha=0.8)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('训练 vs 验证准确率对比', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'{e}' for e in epochs])
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存
    save_path = output_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """绘制混淆矩阵"""

    output_dir = Path(output_dir)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 1. 原始混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=[DISEASE_NAMES[c] for c in DISEASE_CLASSES],
           yticklabels=[DISEASE_NAMES[c] for c in DISEASE_CLASSES],
                cbar_kws={'label': '样本数量'})
    ax1.set_title('混淆矩阵 (原始数量)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('预测类别', fontsize=12)
    ax1.set_ylabel('真实类别', fontsize=12)

    # 2. 归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax2,
                xticklabels=[DISEASE_NAMES[c] for c in DISEASE_CLASSES],
                yticklabels=[DISEASE_NAMES[c] for c in DISEASE_CLASSES],
                cbar_kws={'label': '比例'})
    ax2.set_title('混淆矩阵 (归一化)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('预测类别', fontsize=12)
    ax2.set_ylabel('真实类别', fontsize=12)

    plt.tight_layout()

    # 保存
    save_path = output_dir / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵已保存: {save_path}")
    plt.close()

    return cm

def plot_per_class_metrics(y_true, y_pred, output_dir):
    """绘制每个类别的性能指标"""

    output_dir = Path(output_dir)

    # 生成分类报告
    report = classification_report(y_true, y_pred, target_names=[DISEASE_NAMES[c] for c in DISEASE_CLASSES], output_dict=True)

    # 提取指标
    classes = [DISEASE_NAMES[c] for c in DISEASE_CLASSES]
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    support = [report[c]['support'] for c in classes]

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('各类别性能指标', fontsize=20, fontweight='bold')

    x = np.arange(len(classes))
    width = 0.25

    # 1. Precision, Recall, F1-Score 对比
    axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[0, 0].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('疾病类别', fontsize=12)
    axes[0, 0].set_ylabel('分数', fontsize=12)
    axes[0, 0].set_title('Precision, Recall, F1-Score 对比', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.1])

    # 2. 样本数量分布
    axes[0, 1].bar(x, support, alpha=0.8, color='skyblue')
    axes[0, 1].set_xlabel('疾病类别', fontsize=12)
    axes[0, 1].set_ylabel('样本数量', fontsize=12)
    axes[0, 1].set_title('验证集样本分布', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 在柱子上显示数值
    for i, v in enumerate(support):
        axes[0, 1].text(i, v + max(support)*0.02, str(int(v)), ha='center', va='bottom')

    # 3. F1-Score 排名
    f1_sorted_idx = np.argsort(f1)[::-1]
    axes[1, 0].barh([classes[i] for i in f1_sorted_idx], [f1[i] for i in f1_sorted_idx], alpha=0.8, color='lightgreen')
    axes[1, 0].set_xlabel('F1-Score', fontsize=12)
    axes[1, 0].set_title('F1-Score 排名', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].set_xlim([0, 1.1])

    # 4. 准确率热力图
    accuracy_per_class = [report[c]['recall'] for c in classes]  # Recall 即该类的准确率
    accuracy_matrix = np.array(accuracy_per_class).reshape(1, -1)

    sns.heatmap(accuracy_matrix, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1, 1],
                xticklabels=classes, yticklabels=['准确率'],
                cbar_kws={'label': '准确率'}, vmin=0, vmax=1)
    axes[1, 1].set_title('各类别准确率热力图', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticklabels(classes, rotation=45, ha='right')

    plt.tight_layout()

    # 保存
    save_path = output_dir / 'per_class_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 各类别指标已保存: {save_path}")
    plt.close()

    return report

def generate_performance_report(history, report, cm, output_dir):
    """生成性能报告"""

    output_dir = Path(output_dir)

    # 计算总体指标
    best_epoch = np.argmax(history['val_acc']) + 1
    best_val_acc = max(history['val_acc'])
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]

    # 过拟合分析
    overfit_gap = final_train_acc - final_val_acc

    # 生成报告
    report_text = f"""# 模型性能分析报告

## 训练概况

- **训练轮数**: {len(history['train_loss'])} epochs
- **最佳 Epoch**: {best_epoch}
- **最佳验证准确率**: {best_val_acc:.2f}%

## 最终性能

- **训练准确率**: {final_train_acc:.2f}%
- **验证准确率**: {final_val_acc:.2f}%
- **过拟合程度**: {overfit_gap:.2f}% {'(轻微)' if overfit_gap < 5 else '(中等)' if overfit_gap < 10 else '(严重)'}

## 各类别性能

| 类别 | Precision | Recall | F1-Score | 样本数 |
|------|-----------|--------|----------|--------|
"""

    for cls in DISEASE_CLASSES:
        cls_name = DISEASE_NAMES[cls]
        if cls_name in report:
            p = report[cls_name]['precision']
            r = report[cls_name]['recall']
            f1 = report[cls_name]['f1-score']
            s = int(report[cls_name]['support'])
            report_text += f"| {cls_name} | {p:.4f} | {r:.4f} | {f1:.4f} | {s} |\n"

    report_text += f"""
## 总体指标

- **宏平均 Precision**: {report['macro avg']['precision']:.4f}
- **宏平均 Recall**: {report['macro avg']['recall']:.4f}
- **宏平均 F1-Score**: {report['macro avg']['f1-score']:.4f}
- **加权平均 F1-Score**: {report['weighted avg']['f1-score']:.4f}

## 混淆矩阵分析

### 最容易混淆的类别对

"""

    # 找出最容易混淆的类别
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_normalized, 0)  # 忽略对角线

    confusion_pairs = []
    for i in range(len(DISEASE_CLASSES)):
        for j in range(len(DISEASE_CLASSES)):
            if i != j and cm_normalized[i, j] > 0.1:  # 超过10%的混淆
                confusion_pairs.append((
                    DISEASE_NAMES[DISEASE_CLASSES[i]],
                    DISEASE_NAMES[DISEASE_CLASSES[j]],
                    cm_normalized[i, j]
                ))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    for true_cls, pred_cls, ratio in confusion_pairs[:5]:
        report_text += f"- **{true_cls}** 被误判为 **{pred_cls}**: {ratio:.2%}\n"

    report_text += f"""
## 模型优势

"""

    # 找出表现最好的类别
    f1_scores = [(DISEASE_NAMES[cls], report[DISEASE_NAMES[cls]]['f1-score'])
                 for cls in DISEASE_CLASSES if DISEASE_NAMES[cls] in report]
    f1_scores.sort(key=lambda x: x[1], reverse=True)

    for cls_name, f1 in f1_scores[:3]:
        report_text += f"- **{cls_name}**: F1-Score = {f1:.4f}\n"

    report_text += f"""
## 改进建议

"""

    # 根据结果给出建议
    if overfit_gap > 10:
        report_text += "- ⚠️ 存在明显过拟合，建议增加数据增强或正则化\n"

    if final_val_acc < 70:
        report_text += "- ⚠️ 整体准确率偏低，建议增加训练轮数或调整学习率\n"

    # 找出表现差的类别
    poor_classes = [(cls_name, f1) for cls_name, f1 in f1_scores if f1 < 0.7]
    if poor_classes:
        report_text += f"- ⚠️ 以下类别表现较差，建议增加样本或使用类别权重:\n"
        for cls_name, f1 in poor_classes:
            report_text += f"  - {cls_name}: F1-Score = {f1:.4f}\n"

    if not poor_classes and overfit_gap < 5 and final_val_acc > 80:
        report_text += "- ✅ 模型表现优秀，各项指标均衡\n"

    report_text += f"""
---

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # 保存报告
    report_path = output_dir / 'performance_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"✅ 性能报告已保存: {report_path}")

    return report_text

def plot_dataset_distribution(metadata_path, output_dir):
    """绘制数据集标签分布"""

    output_dir = Path(output_dir)

    print("\n📊 绘制数据集标签分布...")

    # 加载元数据
    metadata = pd.read_csv(metadata_path)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('HAM10000 数据集标签分布分析', fontsize=18, fontweight='bold')

    # 1. 疾病类别分布 - 柱状图
    ax1 = axes[0, 0]
    disease_counts = metadata['dx'].value_counts()
    disease_counts = disease_counts.reindex(DISEASE_CLASSES)

    colors = plt.cm.Set3(range(len(DISEASE_CLASSES)))
    bars = ax1.bar(range(len(DISEASE_CLASSES)), disease_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('疾病类别', fontsize=12)
    ax1.set_ylabel('样本数量', fontsize=12)
    ax1.set_title('疾病类别样本数量分布', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(DISEASE_CLASSES)))
    ax1.set_xticklabels([DISEASE_NAMES[cls] for cls in DISEASE_CLASSES], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # 在柱状图上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. 疾病类别分布 - 饼图
    ax2 = axes[0, 1]
    explode = [0.05 if i == disease_counts.values.argmax() else 0 for i in range(len(DISEASE_CLASSES))]
    wedges, texts, autotexts = ax2.pie(disease_counts.values,
                                        labels=[DISEASE_NAMES[cls] for cls in DISEASE_CLASSES],
                                        autopct='%1.1f%%',
                                        colors=colors,
                                        explode=explode,
                                        startangle=90,
                                        textprops={'fontsize': 9})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax2.set_title('疾病类别占比分布', fontsize=14, fontweight='bold')

    # 3. 类别不平衡分析
    ax3 = axes[1, 0]
    sorted_counts = disease_counts.sort_values(ascending=True)
    y_pos = np.arange(len(sorted_counts))

    bars = ax3.barh(y_pos, sorted_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([DISEASE_NAMES[cls] for cls in sorted_counts.index])
    ax3.set_xlabel('样本数量', fontsize=12)
    ax3.set_title('类别不平衡分析（从少到多）', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(width)}',
                ha='left', va='center', fontsize=10, fontweight='bold')

    # 4. 统计信息表格
    ax4 = axes[1, 1]
    ax4.axis('off')

    total_samples = len(metadata)
    max_class = disease_counts.idxmax()
    min_class = disease_counts.idxmin()
    imbalance_ratio = disease_counts.max() / disease_counts.min()

    stats_text = f"""
数据集统计信息

总样本数: {total_samples:,}
类别数量: {len(DISEASE_CLASSES)}

最多类别: {DISEASE_NAMES[max_class]}
样本数: {disease_counts[max_class]}

最少类别: {DISEASE_NAMES[min_class]}
样本数: {disease_counts[min_class]}

不平衡比例: {imbalance_ratio:.2f}:1
平均样本数: {total_samples/len(DISEASE_CLASSES):.0f}
样本数中位数: {disease_counts.median():.0f}
    """

    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # 保存图表
    save_path = output_dir / 'dataset_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 数据集分布图表已保存: {save_path}")

    plt.close()

    return disease_counts

def visualize_training_results(model_dir='models/skin_lesion_classifier', dataset_dir='datasets/archive (6)'):
    """可视化训练结果"""

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)

    print("="*60)
    print("📊 开始生成训练可视化")
    print("="*60)

    # 加载训练历史
    history_path = model_dir / 'training_history.json'
    if not history_path.exists():
        print(f"❌ 未找到训练历史: {history_path}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    print(f"✅ 加载训练历史: {len(history['train_loss'])} epochs")

    # 创建输出目录
    output_dir = model_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    # 1. 绘制训练曲线
    print("\n📈 生成训练曲线...")
    plot_training_curves(history, output_dir)

    # 2. 绘制数据集标签分布
    metadata_path = dataset_dir / "HAM10000_metadata.csv"
    if metadata_path.exists():
        disease_counts = plot_dataset_distribution(metadata_path, output_dir)

        # 3. 创建综合总结
        print("\n📊 生成综合总结...")
        create_comprehensive_summary(history, disease_counts, output_dir)
    else:
        print(f"⚠️  未找到数据集元数据: {metadata_path}")

    # 4. 加载模型进行评估（如果需要混淆矩阵）
    checkpoint_path = model_dir / 'best_model.pth'
    if checkpoint_path.exists():
        print("\n📦 模型文件存在")
        print("⚠️  详细评估（混淆矩阵）需要重新加载数据，已跳过")
        print("   如需生成，请运行完整评估脚本")

    print("\n" + "="*60)
    print("✅ 可视化完成！")
    print(f"📁 输出目录: {output_dir}")
    print("\n生成的图表:")
    print("  1. training_curves.png - 训练曲线")
    print("  2. dataset_distribution.png - 数据集标签分布")
    print("  3. comprehensive_summary.png - 综合总结")
    print("="*60)

def create_comprehensive_summary(history, disease_counts, output_dir):
    """创建综合总结图表"""

    output_dir = Path(output_dir)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('HAM10000 皮肤病变分类器 - 综合分析报告', fontsize=18, fontweight='bold')

    # 计算关键指标
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    total_samples = disease_counts.sum()

    # 1. 最佳验证准确率卡片
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.text(0.5, 0.7, f'{best_val_acc:.2f}%',
            ha='center', va='center', fontsize=42, fontweight='bold', color='#4CAF50')
    ax1.text(0.5, 0.3, '最佳验证准确率',
            ha='center', va='center', fontsize=13, color='#666')
    ax1.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                edgecolor='#4CAF50', linewidth=3))

    # 2. 最佳轮数卡片
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.text(0.5, 0.7, f'{best_epoch}',
            ha='center', va='center', fontsize=42, fontweight='bold', color='#2196F3')
    ax2.text(0.5, 0.3, '最佳训练轮数',
            ha='center', va='center', fontsize=13, color='#666')
    ax2.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                edgecolor='#2196F3', linewidth=3))

    # 3. 数据集规模卡片
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    ax3.text(0.5, 0.7, f'{total_samples:,}',
            ha='center', va='center', fontsize=42, fontweight='bold', color='#FF9800')
    ax3.text(0.5, 0.3, '总训练样本数',
            ha='center', va='center', fontsize=13, color='#666')
    ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                edgecolor='#FF9800', linewidth=3))

    # 4. 类别分布横向柱状图
    ax4 = fig.add_subplot(gs[1:, :])
    sorted_counts = disease_counts.sort_values(ascending=True)
    y_pos = np.arange(len(sorted_counts))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_counts)))
    bars = ax4.barh(y_pos, sorted_counts.values, color=colors, edgecolor='black', linewidth=2)

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f'{DISEASE_NAMES[cls]}\n({cls})' for cls in sorted_counts.index],
                        fontsize=11)
    ax4.set_xlabel('样本数量', fontsize=13, fontweight='bold')
    ax4.set_title('各类别样本数量分布', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, axis='x')

    # 添加数值标签和百分比
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percentage = (width / total_samples) * 100
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f'  {int(width)} ({percentage:.1f}%)',
                ha='left', va='center', fontsize=11, fontweight='bold')

    # 添加平均线
    avg_samples = total_samples / len(DISEASE_CLASSES)
    ax4.axvline(avg_samples, color='red', linestyle='--', linewidth=2,
               label=f'平均值: {avg_samples:.0f}', alpha=0.7)
    ax4.legend(fontsize=11, loc='lower right')

    plt.tight_layout()

    # 保存图表
    save_path = output_dir / 'comprehensive_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 综合总结图表已保存: {save_path}")

    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='可视化训练结果')
    parser.add_argument('--model-dir', type=str, default='models/skin_lesion_classifier',
                        help='模型目录')
    parser.add_argument('--dataset-dir', type=str, default='datasets/archive (6)',
                        help='数据集目录')

    args = parser.parse_args()

    visualize_training_results(args.model_dir, args.dataset_dir)
