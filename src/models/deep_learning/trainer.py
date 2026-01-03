"""训练器类 - 负责模型训练、验证和测试"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from .mman_model import MMANModel
    from .utils.metrics import MetricsCalculator, plot_training_curves, plot_per_class_metrics
except ImportError:
    from mman_model import MMANModel
    from utils.metrics import MetricsCalculator, plot_training_curves, plot_per_class_metrics

from src.config.config import logger


class Trainer:
    """MMAN模型训练器"""

    def __init__(
        self,
        model: MMANModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        metrics_calculator: Optional[MetricsCalculator] = None,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        """
        Args:
            model: MMAN模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            scheduler: 学习率调度器
            metrics_calculator: 评估指标计算器
            checkpoint_dir: 检查点保存目录
            log_dir: 日志保存目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

        # 评估指标计算器
        self.metrics_calculator = metrics_calculator or MetricsCalculator(
            num_classes=7,
            class_names=[
                '理论讲授型', '启发引导型', '互动导向型', '逻辑推导型',
                '题目驱动型', '情感表达型', '耐心细致型'
            ]
        )

        # 目录设置
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 训练历史
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.current_epoch = 0

    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个epoch

        Returns:
            (avg_loss, avg_acc): 平均损失和平均准确率
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch in pbar:
            # 将数据移到设备
            features = {
                k: v.to(self.device) for k, v in batch['features'].items()
            }
            labels = batch['labels'].to(self.device)
            rule_features = batch.get('rule_features')
            if rule_features is not None:
                rule_features = rule_features.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(features, rule_features)
            logits = outputs['logits']
            predictions = outputs['predictions']

            # 计算损失
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict[str, float]]:
        """
        验证模型

        Returns:
            (avg_loss, avg_acc, metrics): 平均损失、平均准确率和详细指标
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for batch in pbar:
            # 将数据移到设备
            features = {
                k: v.to(self.device) for k, v in batch['features'].items()
            }
            labels = batch['labels'].to(self.device)
            rule_features = batch.get('rule_features')
            if rule_features is not None:
                rule_features = rule_features.to(self.device)

            # 前向传播
            outputs = self.model(features, rule_features)
            logits = outputs['logits']
            predictions = outputs['predictions']
            probabilities = outputs['probabilities']

            # 计算损失
            loss = self.criterion(logits, labels)

            # 记录
            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_preds, all_probs
        )
        avg_acc = metrics['accuracy']

        return avg_loss, avg_acc, metrics

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_best_only: bool = True
    ):
        """
        训练模型

        Args:
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_best_only: 是否只保存最佳模型
        """
        logger.info(f"开始训练 - {num_epochs} epochs")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练样本数: {len(self.train_loader.dataset)}")
        logger.info(f"验证样本数: {len(self.val_loader.dataset)}")

        best_epoch = 0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # 验证
            val_loss, val_acc, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 日志
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )

            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # 保存最佳模型
                self.save_checkpoint(
                    epoch,
                    is_best=True,
                    metrics=val_metrics
                )
                logger.info(f"✓ 保存最佳模型 (Acc: {val_acc:.4f})")
            else:
                patience_counter += 1

            # 定期保存检查点
            if not save_best_only and (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # 早停
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"Best epoch: {best_epoch + 1}, Best val acc: {self.best_val_acc:.4f}")
                break

        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"\n训练完成!")
        logger.info(f"总耗时: {total_time / 60:.2f} 分钟")
        logger.info(f"最佳验证准确率: {self.best_val_acc:.4f} (Epoch {best_epoch + 1})")

        # 绘制训练曲线
        self.plot_training_history()

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        保存检查点

        Args:
            epoch: 当前epoch
            is_best: 是否为最佳模型
            metrics: 评估指标
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存检查点
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'

        torch.save(checkpoint, checkpoint_path)

        # 保存指标到JSON
        if metrics and is_best:
            metrics_path = self.checkpoint_dir / 'best_metrics.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accs = checkpoint.get('val_accs', [])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.current_epoch = checkpoint.get('epoch', 0)

        logger.info(f"检查点加载成功: {checkpoint_path}")
        logger.info(f"Epoch: {self.current_epoch}, Best Val Acc: {self.best_val_acc:.4f}")

    def plot_training_history(self):
        """绘制训练历史"""
        if len(self.train_losses) > 0:
            save_path = self.log_dir / 'training_curves.png'
            plot_training_curves(
                self.train_losses,
                self.val_losses,
                self.train_accs,
                self.val_accs,
                save_path=str(save_path)
            )

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        测试模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            metrics: 测试指标
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        logger.info("开始测试...")

        for batch in tqdm(test_loader, desc="Testing"):
            features = {
                k: v.to(self.device) for k, v in batch['features'].items()
            }
            labels = batch['labels'].to(self.device)
            rule_features = batch.get('rule_features')
            if rule_features is not None:
                rule_features = rule_features.to(self.device)

            outputs = self.model(features, rule_features)
            predictions = outputs['predictions']
            probabilities = outputs['probabilities']

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_preds, all_probs
        )

        # 打印报告
        logger.info("\n" + "=" * 80)
        logger.info("测试结果")
        logger.info("=" * 80)
        logger.info(self.metrics_calculator.get_classification_report(all_labels, all_preds))
        logger.info("=" * 80)

        # 绘制混淆矩阵
        cm_path = self.log_dir / 'confusion_matrix.png'
        self.metrics_calculator.plot_confusion_matrix(
            all_labels, all_preds,
            normalize='true',
            save_path=str(cm_path)
        )

        # 绘制每个类别的指标
        per_class_path = self.log_dir / 'per_class_metrics.png'
        plot_per_class_metrics(
            metrics,
            self.metrics_calculator.class_names,
            save_path=str(per_class_path)
        )

        # 保存测试指标
        test_metrics_path = self.log_dir / 'test_metrics.json'
        with open(test_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics
