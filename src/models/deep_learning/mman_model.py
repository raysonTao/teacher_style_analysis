"""MMAN模型：Multi-Modal Attention Network

基于Transformer和LSTM的混合架构，用于教师教学风格分类
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from .components import (
        ModalityEncoder,
        TransformerEncoder,
        BiLSTMEncoder,
        AttentionPooling
    )
    from .config import ModelConfig, DEFAULT_CONFIG
except ImportError:
    # 直接导入（用于测试）
    from components import (
        ModalityEncoder,
        TransformerEncoder,
        BiLSTMEncoder,
        AttentionPooling
    )
    from config import ModelConfig, DEFAULT_CONFIG


class MMANModel(nn.Module):
    """Multi-Modal Attention Network for Teacher Style Classification

    架构流程:
    1. 模态特定编码: 将各模态特征映射到统一嵌入空间
    2. 跨模态Transformer: 捕获模态间的交互和依赖
    3. 时序建模LSTM: 建模时序特征
    4. 注意力池化: 聚合序列信息
    5. 规则特征融合: 可选地融合规则系统输出
    6. 分类头: 产生7种教学风格的概率分布
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()

        self.config = config if config is not None else DEFAULT_CONFIG

        # 1. 模态特定编码器
        self.modality_encoders = nn.ModuleDict({
            'video': ModalityEncoder(
                self.config.input_dims['video'],
                self.config.embedding_dim,
                self.config.transformer_dropout
            ),
            'audio': ModalityEncoder(
                self.config.input_dims['audio'],
                self.config.embedding_dim,
                self.config.transformer_dropout
            ),
            'text': ModalityEncoder(
                self.config.input_dims['text'],
                self.config.embedding_dim,
                self.config.transformer_dropout
            )
        })

        # 2. 跨模态Transformer编码器
        self.transformer = TransformerEncoder(
            embed_dim=self.config.embedding_dim,
            num_layers=self.config.transformer_layers,
            num_heads=self.config.transformer_heads,
            ff_dim=self.config.transformer_ff_dim,
            dropout=self.config.transformer_dropout
        )

        # 3. 双向LSTM层
        self.lstm = BiLSTMEncoder(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_layers,
            dropout=self.config.lstm_dropout,
            bidirectional=self.config.lstm_bidirectional
        )

        # 4. 注意力池化层
        self.attention_pooling = AttentionPooling(
            input_dim=self.config.lstm_output_dim,
            hidden_dim=self.config.attention_hidden_dim
        )

        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(
                self.config.classifier_input_dim,
                self.config.classifier_hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(
                self.config.classifier_hidden_dim,
                self.config.num_classes
            )
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        rule_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            features: 多模态特征字典
                - 'video': [batch_size, video_dim]
                - 'audio': [batch_size, audio_dim]
                - 'text': [batch_size, text_dim]
            rule_features: [batch_size, rule_dim] 规则系统输出 (可选)
            return_attention: 是否返回注意力权重

        Returns:
            output_dict: 包含以下键的字典
                - 'logits': [batch_size, num_classes] 未归一化的分类分数
                - 'probabilities': [batch_size, num_classes] 概率分布
                - 'predictions': [batch_size] 预测的类别索引
                - 'transformer_attention': (可选) Transformer注意力权重
                - 'pooling_attention': (可选) 池化注意力权重
        """
        batch_size = features['video'].size(0)

        # 1. 模态特定编码
        encoded_modalities = []
        for modality_name in ['video', 'audio', 'text']:
            encoded = self.modality_encoders[modality_name](features[modality_name])
            encoded_modalities.append(encoded)

        # 堆叠为序列 [batch_size, 3, embedding_dim]
        multimodal_sequence = torch.stack(encoded_modalities, dim=1)

        # 2. 跨模态Transformer编码
        transformer_output, transformer_attention = self.transformer(multimodal_sequence)
        # transformer_output: [batch_size, 3, embedding_dim]

        # 3. LSTM时序建模
        lstm_output, (h_n, c_n) = self.lstm(transformer_output)
        # lstm_output: [batch_size, 3, lstm_output_dim]

        # 4. 注意力池化
        pooled_output, pooling_attention = self.attention_pooling(lstm_output)
        # pooled_output: [batch_size, lstm_output_dim]

        # 5. 融合规则特征（如果提供）
        if self.config.use_rule_features and rule_features is not None:
            pooled_output = torch.cat([pooled_output, rule_features], dim=-1)

        # 6. 分类
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        # 构建输出字典
        output_dict = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions
        }

        if return_attention:
            output_dict['transformer_attention'] = transformer_attention
            output_dict['pooling_attention'] = pooling_attention

        return output_dict

    def get_embedding(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        获取特征嵌入（用于可视化或下游任务）

        Args:
            features: 多模态特征字典

        Returns:
            embedding: [batch_size, lstm_output_dim] 特征嵌入向量
        """
        # 前向传播直到池化层
        encoded_modalities = []
        for modality_name in ['video', 'audio', 'text']:
            encoded = self.modality_encoders[modality_name](features[modality_name])
            encoded_modalities.append(encoded)

        multimodal_sequence = torch.stack(encoded_modalities, dim=1)
        transformer_output, _ = self.transformer(multimodal_sequence)
        lstm_output, _ = self.lstm(transformer_output)
        pooled_output, _ = self.attention_pooling(lstm_output)

        return pooled_output

    @property
    def num_parameters(self) -> int:
        """返回模型总参数量"""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_summary(self):
        """打印模型结构摘要"""
        print("=" * 80)
        print("MMAN Model Summary")
        print("=" * 80)
        print(f"Input dimensions:")
        for modality, dim in self.config.input_dims.items():
            print(f"  - {modality}: {dim}")
        print(f"Embedding dimension: {self.config.embedding_dim}")
        print(f"Transformer: {self.config.transformer_layers} layers, {self.config.transformer_heads} heads")
        print(f"LSTM: {self.config.lstm_layers} layers, hidden={self.config.lstm_hidden_dim}, bidirectional={self.config.lstm_bidirectional}")
        print(f"Output classes: {self.config.num_classes}")
        print(f"Use rule features: {self.config.use_rule_features}")
        print(f"\nTotal parameters: {self.num_parameters:,}")
        print(f"Trainable parameters: {self.num_trainable_parameters:,}")
        print("=" * 80)


def create_model(config_name: str = 'default') -> MMANModel:
    """
    创建MMAN模型的便捷函数

    Args:
        config_name: 配置名称 ('default', 'lightweight', 'high_accuracy')

    Returns:
        model: MMAN模型实例
    """
    try:
        from .config import DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG, HIGH_ACCURACY_CONFIG
    except ImportError:
        from config import DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG, HIGH_ACCURACY_CONFIG

    config_map = {
        'default': DEFAULT_CONFIG,
        'lightweight': LIGHTWEIGHT_CONFIG,
        'high_accuracy': HIGH_ACCURACY_CONFIG
    }

    if config_name not in config_map:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(config_map.keys())}")

    config = config_map[config_name]
    model = MMANModel(config)

    return model


if __name__ == '__main__':
    # 测试代码
    print("Testing MMAN Model...")

    # 创建模型
    model = create_model('default')
    model.print_model_summary()

    # 创建测试输入
    batch_size = 4
    test_features = {
        'video': torch.randn(batch_size, 20),
        'audio': torch.randn(batch_size, 15),
        'text': torch.randn(batch_size, 25)
    }
    test_rule_features = torch.randn(batch_size, 7)

    # 前向传播
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(test_features, test_rule_features, return_attention=True)

    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probabilities shape: {output['probabilities'].shape}")
    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Sample predictions: {output['predictions']}")
    print(f"Sample probabilities:\n{output['probabilities']}")

    print("\nModel test passed!")
