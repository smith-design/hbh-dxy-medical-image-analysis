"""
训练可视化脚本
监控和可视化模型训练过程
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_training_logs(log_dir):
    """加载训练日志"""

    log_file = Path(log_dir) / "trainer_state.json"

    if not log_file.exists():
        print(f"❌ 未找到训练日志: {log_file}")
        return None

    with open(log_file, 'r') as f:
        state = json.load(f)

    return state

def plot_loss_curves(state, output_dir):
    """绘制损失曲线"""

    log_history = state.get('log_history', [])

    if not log_history:
        print("⚠️  没有训练日志数据")
        return

    # 提取训练和验证数据
    train_data = []
    eval_data = []

    for entry in log_history:
        if 'loss' in entry:
            train_data.append({
                'step': entry.get('step', 0),
                'epoch': entry.get('epoch', 0),
         'loss': entry['loss']
        })
        if 'eval_loss' in entry:
            eval_data.append({
                'step': entry.get('step', 0),
                'epoch': entry.get('epoch', 0),
                'eval_loss': entry['eval_loss']
            })

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('模型训练可视化分析', fontsize=16, fontweight='bold')

    # 1. 训练损失曲线
    if train_data:
        df_train = pd.DataFrame(train_data)
        axes[0, 0].plot(df_train['step'], df_train['loss'],
                       linewidth=2, marker='o', markersize=4, label='训练损失')
        axes[0, 0].set_xlabel('训练步数 (Steps)', fontsize=12)
        axes[0, 0].set_ylabel('损失值 (Loss)', fontsize=12)
        axes[0, 0].set_title('训练损失曲线', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

    # 2. 验证损失曲线
    if eval_data:
        df_eval = pd.DataFrame(eval_data)
        axes[0, 1].plot(df_eval['step'], df_eval['eval_loss'],
                       linewidth=2, marker='s', markersize=4,
                       color='orange', label='验证损失')
        axes[0, 1].set_xlabel('训练步数 (Steps)', fontsize=12)
        axes[0, 1].set_ylabel('损失值 (Loss)', fontsize=12)
        axes[0, 1].set_title('验证损失曲线', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

    # 3. 训练和验证损失对比
    if train_data and eval_data:
        axes[1, 0].plot(df_train['step'], df_train['loss'],
                       linewidth=2, label='训练损失', alpha=0.8)
        axes[1, 0].plot(df_eval['step'], df_eval['eval_loss'],
                       linewidth=2, label='验证损失', alpha=0.8)
        axes[1, 0].set_xlabel('训练步数 (Steps)', fontsize=12)
        axes[1, 0].set_ylabel('损失值 (Loss)', fontsize=12)
        axes[1, 0].set_title('训练 vs 验证损失对比', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

     # 计算过拟合指标
        if len(df_eval) > 0:
            last_train_loss = df_train['loss'].iloc[-1]
            last_eval_loss = df_eval['eval_loss'].iloc[-1]
            gap = last_eval_loss - last_train_loss
            axes[1, 0].text(0.02, 0.98,
                          f'最终训练损失: {last_train_loss:.4f}\n'
                          f'最终验证损失: {last_eval_loss:.4f}\n'
                          f'损失差距: {gap:.4f}',
                          transform=axes[1, 0].transAxes,
                          verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          fontsize=10)

    # 4. 学习率变化（如果有）
    lr_data = [entry for entry in log_history if 'learning_rate' in entry]
    if lr_data:
        df_lr = pd.DataFrame(lr_data)
        axes[1, 1].plot(df_lr['step'], df_lr['learning_rate'],
                       linewidth=2, color='green', marker='d', markersize=4)
        axes[1, 1].set_xlabel('训练步数 (Steps)', fontsize=12)
        axes[1, 1].set_ylabel('学习率 (Learning Rate)', fontsize=12)
        axes[1, 1].set_title('学习率变化曲线', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    else:
        axes[1, 1].text(0.5, 0.5, '无学习率数据',
                       ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')

    plt.tight_layout()

    # 保存图表
    output_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bnches='tight')
    print(f"✅ 训练曲线已保存: {output_path}")

    plt.close()

def plot_epoch_summary(state, output_dir):
    """绘制每个 epoch 的汇总统计"""

    log_history = state.get('log_history', [])

    # 按 epoch 分组
    epoch_data = {}
    for entry in log_history:
        if 'epoch' in entry:
            epoch = entry['epoch']
            if epoch not in epoch_data:
                epoch_data[epoch] = {'train_loss': [], 'eval_loss': None}
          if 'loss' in entry:
                epoch_data[epoch]['train_loss'].append(entry['loss'])
            if 'eval_loss' in entry:
                epoch_data[epoch]['eval_loss'] = entry['eval_loss']

    if not epoch_data:
        print("⚠️  没有 epoch 数据")
        return

    # 准备数据
    epochs = sorted(epoch_data.keys())
    avg_train_loss = [np.mean(epoch_data[e]['train_loss']) if epoch_data[e]['train_loss'] else 0
                     for e in epochs]
    eval_loss = [epoch_data[e]['eval_loss'] if epoch_data[e]['eval_loss'] else 0
                for e in epochs]

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange((epochs))
    width = 0.35

    bars1 = ax.bar(x - width/2, avg_train_loss, width, label='平均训练损失', alpha=0.8)
    bars2 = ax.bar(x + width/2, eval_loss, width, label='验证损失', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('损失值 (Loss)', fontsize=12)
    ax.set_title('每个 Epoch 的损失对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Epoch {int(e)}' for e in epochs])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存图表
    output_path = Path(output_dir) / 'epoch_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Epoch 汇总已保存: {output_path}")

    plt.close()

def generate_training_report(state, output_dir):
    """生成训练报告"""

    log_history = state.get('log_history', [])

    if not log_history:
        return

    # 提取关键指标
    train_losses = [e['loss'] for e in log_history if 'loss' in e]
    eval_losses = [e['eval_loss'] for e in log_history if 'eval_loss' in e]

    report = f"""
# 训练报告

## 训练配置
- 总训练步数: {state.get('global_step', 'N/A')}
- 总 Epoch 数: {state.get('epoch', 'N/A')}
- 最佳模型检查点: {state.get('best_model_checkpoint', 'N/A')}

## 训练指标

### 训练损失
- 初始损失: {train_losses[0]:.4f}
- 最终损失: {train_losses[-1]:.4f}
- 最低损失: {min(train_losses):.4f}
- 平均损失: {np.mean(train_losses):.4f}
- 损失下降: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%

### 验证损失
"""

    if eval_losses:
        report += f"""- 初始验证损失: {eval_losses[0]:.4f}
- 最终验证损失: {eval_losses[-1]:.4f}
- 最低验证损失: {min(eval_losses):.4f}
- 平均验证损失: {np.mean(eval_losses):.4f}

### 过拟合分析
- 训练-验证损失差距: {(eval_losses[-1] - train_losses[-1]):.4f}
- 过拟合程度: {'轻微' if abs(eval_losses[-1] - train_losses[-1]) < 0.1 else '中等' if abs(eval_losses[-1] - train_losses[-1]) < 0.3 else '严重'}
"""

    report += f"""
## 训练稳定性
- 训练损失标准差: {np.std(train_losses):.4f}
- 训练损失变异系数: {(np.std(train_losses) / np.mean(train_losses)):.4f}

## 建议
"""

    # 根据指标给出建议
    if eval_losses and eval_losses[-1] > train_losses[-1] + 0.3:
        report += "- ⚠️ 检测到明显过拟合，建议增加正则化或减少训练轮数\n"

    if train_losses[-1] > train_losses[0] * 0.8:
        report += "- ⚠️ 损失下降不明显，建议调整学习率或增加训练轮数\n"

    if np.std(train_losses) > np.mean(train_losses) * 0.5:
        report += "- ⚠️ 训练不稳定，建议降低学习率或调整批次大小\n"

    if not (eval_losses and eval_losses[-1] > train_losses[-1] + 0.3) and \
       not (train_losses[-1] > train_losses[0] * 0.8):
        report += "- ✅ 训练效果良好，模型收敛正常\n"

    # 保存报告
    output_path = Path(output_dir) / 'training_report.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 训练报告已保存: {output_path}")

def main():
    """主函数"""

    # 路径配置
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / "models" / "qwen2vl_ham10000_lora"
    output_dir = base_dir / "models" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📊 开始生成训练可视化...")

    # 加载训练日志
    state = load_training_logs(model_dir)

    if state is None:
        print("❌ 无法加载训练日志")
        return

    # 生成可视化
    print("\n📈 绘制训练曲线...")
    plot_loss_curves(state, output_dir)

    print("\n📊 绘制 Epoch 汇总...")
    plot_epoch_summary(state, output_dir)

    print("\n📝 生成训练报告...")
    generate_training_report(state, output_dir)

    print(f"\n✅ 所有可视化已完成！")
    print(f"📁 输出目录: {output_dir}")

if __name__ == "__main__":
    main()
