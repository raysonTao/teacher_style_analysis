"""PyTorch数据集类和数据加载工具"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.config.config import logger


class TeacherStyleDataset(Dataset):
    """教师风格数据集类"""

    # 7种教学风格标签
    STYLE_LABELS = [
        '理论讲授型',  # 0
        '启发引导型',  # 1
        '互动导向型',  # 2
        '逻辑推导型',  # 3
        '题目驱动型',  # 4
        '情感表达型',  # 5
        '耐心细致型'   # 6
    ]

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        use_rule_features: bool = True,
        transform: Optional[callable] = None
    ):
        """
        Args:
            data_path: 数据路径（JSON文件或目录）
            split: 数据集划分 ('train', 'val', 'test')
            use_rule_features: 是否使用规则系统特征
            transform: 数据转换函数
        """
        self.data_path = Path(data_path)
        self.split = split
        self.use_rule_features = use_rule_features
        self.transform = transform

        # 加载数据
        self.data = self._load_data()

        logger.info(f"Loaded {len(self.data)} samples for {split} set")

    def _load_data(self) -> List[Dict]:
        """加载数据"""
        # 如果是JSON文件，直接加载
        if self.data_path.is_file() and self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)

            # 如果数据有split字段，过滤对应的split
            if 'split' in all_data[0]:
                data = [item for item in all_data if item['split'] == self.split]
            else:
                data = all_data

        # 如果是目录，加载该目录下所有JSON文件
        elif self.data_path.is_dir():
            data = []
            json_files = list(self.data_path.glob(f"{self.split}_*.json"))
            if not json_files:
                json_files = list(self.data_path.glob("*.json"))

            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            sample: 包含以下键的字典
                - 'features': {'video': Tensor, 'audio': Tensor, 'text': Tensor}
                - 'rule_features': Tensor (如果use_rule_features=True)
                - 'label': Tensor (标量)
                - 'label_name': str
                - 'sample_id': str
        """
        item = self.data[idx]

        # 提取特征
        features = {
            'video': torch.tensor(item['video_features'], dtype=torch.float32),
            'audio': torch.tensor(item['audio_features'], dtype=torch.float32),
            'text': torch.tensor(item['text_features'], dtype=torch.float32)
        }

        # 提取标签
        if isinstance(item['label'], str):
            # 如果标签是字符串，转换为索引
            label = self.STYLE_LABELS.index(item['label'])
            label_name = item['label']
        else:
            # 如果标签是索引
            label = item['label']
            label_name = self.STYLE_LABELS[label]

        # 构建样本
        sample = {
            'features': features,
            'label': torch.tensor(label, dtype=torch.long),
            'label_name': label_name,
            'sample_id': item.get('sample_id', f"{self.split}_{idx}")
        }

        # 添加规则特征（如果有）
        if self.use_rule_features and 'rule_features' in item:
            sample['rule_features'] = torch.tensor(item['rule_features'], dtype=torch.float32)
        elif self.use_rule_features:
            # 如果没有规则特征，用零填充
            sample['rule_features'] = torch.zeros(7, dtype=torch.float32)

        # 应用数据转换
        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        批处理函数

        Args:
            batch: 样本列表

        Returns:
            batched: 批次化的数据字典
        """
        # 分离各个字段
        video_features = torch.stack([item['features']['video'] for item in batch])
        audio_features = torch.stack([item['features']['audio'] for item in batch])
        text_features = torch.stack([item['features']['text'] for item in batch])

        labels = torch.stack([item['label'] for item in batch])
        label_names = [item['label_name'] for item in batch]
        sample_ids = [item['sample_id'] for item in batch]

        batched = {
            'features': {
                'video': video_features,
                'audio': audio_features,
                'text': text_features
            },
            'labels': labels,
            'label_names': label_names,
            'sample_ids': sample_ids
        }

        # 添加规则特征（如果有）
        if 'rule_features' in batch[0]:
            rule_features = torch.stack([item['rule_features'] for item in batch])
            batched['rule_features'] = rule_features

        return batched


def create_data_loaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_rule_features: bool = True,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器

    Args:
        data_path: 数据路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        use_rule_features: 是否使用规则特征
        val_split: 验证集比例
        test_split: 测试集比例

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 创建数据集
    train_dataset = TeacherStyleDataset(
        data_path,
        split='train',
        use_rule_features=use_rule_features
    )

    val_dataset = TeacherStyleDataset(
        data_path,
        split='val',
        use_rule_features=use_rule_features
    )

    test_dataset = TeacherStyleDataset(
        data_path,
        split='test',
        use_rule_features=use_rule_features
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=TeacherStyleDataset.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=TeacherStyleDataset.collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=TeacherStyleDataset.collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# 模拟数据生成（用于测试，稍后会替换为真实数据）
def generate_synthetic_data(
    num_samples: int = 1000,
    output_path: str = None,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> str:
    """
    生成合成数据用于测试

    Args:
        num_samples: 样本数量
        output_path: 输出路径
        val_split: 验证集比例
        test_split: 测试集比例

    Returns:
        output_path: 数据保存路径
    """
    if output_path is None:
        output_path = "/mnt/data/teacher_style_analysis/synthetic_data.json"

    np.random.seed(42)

    # 计算各split的样本数
    num_val = int(num_samples * val_split)
    num_test = int(num_samples * test_split)
    num_train = num_samples - num_val - num_test

    data = []

    for i in range(num_samples):
        # 确定split
        if i < num_train:
            split = 'train'
        elif i < num_train + num_val:
            split = 'val'
        else:
            split = 'test'

        # 随机生成标签
        label = np.random.randint(0, 7)

        # 生成特征（基于标签添加一些相关性）
        video_features = np.random.randn(20).astype(float) + label * 0.1
        audio_features = np.random.randn(15).astype(float) + label * 0.1
        text_features = np.random.randn(25).astype(float) + label * 0.1

        # 生成规则特征（基于标签的one-hot + 噪声）
        rule_features = np.random.rand(7).astype(float) * 0.2
        rule_features[label] += 0.6

        sample = {
            'sample_id': f"sample_{i:04d}",
            'split': split,
            'label': int(label),
            'video_features': video_features.tolist(),
            'audio_features': audio_features.tolist(),
            'text_features': text_features.tolist(),
            'rule_features': rule_features.tolist()
        }

        data.append(sample)

    # 保存数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Generated {num_samples} synthetic samples")
    logger.info(f"  Train: {num_train}, Val: {num_val}, Test: {num_test}")
    logger.info(f"  Saved to: {output_path}")

    return output_path


if __name__ == '__main__':
    print("Testing TeacherStyleDataset...")

    # 生成合成数据
    data_path = generate_synthetic_data(num_samples=100)

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path,
        batch_size=8,
        num_workers=0  # 测试时不使用多线程
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 测试一个批次
    print("\nTesting a batch...")
    batch = next(iter(train_loader))

    print(f"Video features shape: {batch['features']['video'].shape}")
    print(f"Audio features shape: {batch['features']['audio'].shape}")
    print(f"Text features shape: {batch['features']['text'].shape}")
    print(f"Rule features shape: {batch['rule_features'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Sample labels: {batch['labels']}")
    print(f"Sample label names: {batch['label_names']}")

    print("\nDataset test passed!")
