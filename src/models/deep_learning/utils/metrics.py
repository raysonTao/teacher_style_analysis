"""评估指标计算工具"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self, num_classes: int = 7, class_names: Optional[List[str]] = None):
        """
        Args:
            num_classes: 类别数量
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        计算所有评估指标

        Args:
            y_true: 真实标签 [N]
            y_pred: 预测标签 [N]
            y_prob: 预测概率 [N, num_classes] (可选)

        Returns:
            metrics: 包含所有指标的字典
        """
        metrics = {}

        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 每个类别的指标
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(self.num_classes)))
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(self.num_classes)))
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(self.num_classes)))

        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = float(precision_per_class[i])
            metrics[f'recall_{class_name}'] = float(recall_per_class[i])
            metrics[f'f1_{class_name}'] = float(f1_per_class[i])

        # AUC (如果提供了概率)
        if y_prob is not None:
            try:
                # 多分类AUC (one-vs-rest)
                metrics['auc_ovr'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro'
                )
                metrics['auc_ovo'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovo', average='macro'
                )
            except ValueError:
                # 如果某些类别没有样本，跳过AUC计算
                pass

        return metrics

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        计算混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            normalize: 归一化方式 ('true', 'pred', 'all', None)

        Returns:
            cm: 混淆矩阵
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()

        return cm

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        生成分类报告

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            report: 分类报告字符串
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            labels=list(range(self.num_classes)),
            zero_division=0
        )

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            normalize: 归一化方式
            save_path: 保存路径
            figsize: 图像大小
        """
        cm = self.get_confusion_matrix(y_true, y_pred, normalize)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

        plt.close()

    @staticmethod
    def calculate_top_k_accuracy(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        k: int = 3
    ) -> float:
        """
        计算Top-K准确率

        Args:
            y_true: 真实标签 [N]
            y_prob: 预测概率 [N, num_classes]
            k: Top K

        Returns:
            top_k_acc: Top-K准确率
        """
        top_k_pred = np.argsort(y_prob, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_pred[i]:
                correct += 1
        return correct / len(y_true)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    绘制训练曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_path: 保存路径
        figsize: 图像大小
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")

    plt.close()


def plot_per_class_metrics(
    metrics: Dict[str, float],
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    绘制每个类别的指标柱状图

    Args:
        metrics: 指标字典
        class_names: 类别名称列表
        save_path: 保存路径
        figsize: 图像大小
    """
    # 提取每个类别的precision, recall, f1
    precision = [metrics[f'precision_{name}'] for name in class_names]
    recall = [metrics[f'recall_{name}'] for name in class_names]
    f1 = [metrics[f'f1_{name}'] for name in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to: {save_path}")

    plt.close()


if __name__ == '__main__':
    print("Testing MetricsCalculator...")

    # 模拟数据
    np.random.seed(42)
    n_samples = 100
    n_classes = 7

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # 归一化为概率

    class_names = [
        '理论讲授型', '启发引导型', '互动导向型', '逻辑推导型',
        '题目驱动型', '情感表达型', '耐心细致型'
    ]

    # 创建计算器
    calculator = MetricsCalculator(num_classes=n_classes, class_names=class_names)

    # 计算指标
    metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)

    print("\n评估指标:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    if 'auc_ovr' in metrics:
        print(f"AUC (OvR): {metrics['auc_ovr']:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(calculator.get_classification_report(y_true, y_pred))

    # Top-K准确率
    top3_acc = MetricsCalculator.calculate_top_k_accuracy(y_true, y_prob, k=3)
    print(f"\nTop-3 Accuracy: {top3_acc:.4f}")

    print("\nMetrics calculator test passed!")
