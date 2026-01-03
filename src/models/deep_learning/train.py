"""训练脚本 - MMAN模型训练主入口"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.deep_learning.mman_model import MMANModel, create_model
from src.models.deep_learning.config import ModelConfig, DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG, HIGH_ACCURACY_CONFIG
from src.models.deep_learning.dataset import create_data_loaders, generate_synthetic_data
from src.models.deep_learning.trainer import Trainer
from src.models.deep_learning.utils.metrics import MetricsCalculator
from src.config.config import logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练MMAN教师风格分类模型')

    # 数据参数
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据路径 (JSON文件或目录)')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='使用合成数据进行测试')
    parser.add_argument('--num_synthetic', type=int, default=1000,
                        help='合成数据样本数')

    # 模型参数
    parser.add_argument('--model_config', type=str, default='default',
                        choices=['default', 'lightweight', 'high_accuracy'],
                        help='模型配置')
    parser.add_argument('--use_rule_features', action='store_true', default=True,
                        help='是否使用规则系统特征')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine', 'plateau'],
                        help='学习率调度器')

    # 早停和检查点
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='训练设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 测试参数
    parser.add_argument('--test_only', action='store_true',
                        help='仅测试模型')
    parser.add_argument('--test_checkpoint', type=str, default=None,
                        help='测试用的检查点路径')

    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_optimizer(model: nn.Module, args) -> optim.Optimizer:
    """创建优化器"""
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    return optimizer


def create_scheduler(optimizer: optim.Optimizer, args, num_epochs: int):
    """创建学习率调度器"""
    if args.scheduler == 'none':
        return None
    elif args.scheduler == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1
        )
    elif args.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
    elif args.scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 准备数据
    if args.use_synthetic or args.data_path is None:
        logger.info("生成合成数据用于测试...")
        data_path = generate_synthetic_data(
            num_samples=args.num_synthetic,
            output_path="/mnt/data/teacher_style_analysis/synthetic_data.json"
        )
    else:
        data_path = args.data_path

    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path=data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_rule_features=args.use_rule_features
    )

    logger.info(f"训练样本: {len(train_loader.dataset)}")
    logger.info(f"验证样本: {len(val_loader.dataset)}")
    logger.info(f"测试样本: {len(test_loader.dataset)}")

    # 创建模型
    logger.info(f"创建模型: {args.model_config}")
    model = create_model(args.model_config)
    model.print_model_summary()

    # 创建优化器
    optimizer = create_optimizer(model, args)
    logger.info(f"优化器: {args.optimizer}, 学习率: {args.lr}")

    # 创建调度器
    scheduler = create_scheduler(optimizer, args, args.num_epochs)
    if scheduler:
        logger.info(f"学习率调度器: {args.scheduler}")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 评估指标计算器
    metrics_calculator = MetricsCalculator(
        num_classes=7,
        class_names=[
            '理论讲授型', '启发引导型', '互动导向型', '逻辑推导型',
            '题目驱动型', '情感表达型', '耐心细致型'
        ]
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        metrics_calculator=metrics_calculator,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # 从检查点恢复
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 仅测试模式
    if args.test_only:
        if args.test_checkpoint:
            logger.info(f"加载测试检查点: {args.test_checkpoint}")
            trainer.load_checkpoint(args.test_checkpoint)
        elif not args.resume:
            logger.warning("测试模式但未指定检查点，使用未训练的模型")

        logger.info("=" * 80)
        logger.info("测试模式")
        logger.info("=" * 80)
        test_metrics = trainer.test(test_loader)

        logger.info(f"\n测试准确率: {test_metrics['accuracy']:.4f}")
        logger.info(f"测试F1 (macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"测试F1 (weighted): {test_metrics['f1_weighted']:.4f}")

        return

    # 开始训练
    logger.info("=" * 80)
    logger.info("开始训练")
    logger.info("=" * 80)

    trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping,
        save_best_only=True
    )

    # 加载最佳模型并测试
    logger.info("\n" + "=" * 80)
    logger.info("使用最佳模型进行测试")
    logger.info("=" * 80)

    best_checkpoint = Path(args.checkpoint_dir) / 'best_model.pth'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))
        test_metrics = trainer.test(test_loader)

        logger.info(f"\n最终测试结果:")
        logger.info(f"  测试准确率: {test_metrics['accuracy']:.4f}")
        logger.info(f"  测试F1 (macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"  测试F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    else:
        logger.warning("未找到最佳模型检查点")

    logger.info("\n训练完成!")


if __name__ == '__main__':
    main()
