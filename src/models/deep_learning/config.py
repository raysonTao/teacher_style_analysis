"""深度学习模型配置文件"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    """MMAN模型配置类"""

    # 输入特征维度
    input_dims: Dict[str, int] = None

    # 嵌入维度
    embedding_dim: int = 128

    # Transformer配置
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_dim: int = 512
    transformer_dropout: float = 0.1

    # LSTM配置
    lstm_hidden_dim: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True

    # 注意力配置
    attention_hidden_dim: int = 64

    # 分类配置
    num_classes: int = 7

    # 规则系统集成
    use_rule_features: bool = True
    rule_feature_dim: int = 7

    # 分类头配置
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.3

    def __post_init__(self):
        """后初始化，设置默认输入维度"""
        if self.input_dims is None:
            self.input_dims = {
                'video': 20,
                'audio': 15,
                'text': 25
            }

    @property
    def total_input_dim(self) -> int:
        """计算总输入维度"""
        return sum(self.input_dims.values())

    @property
    def lstm_output_dim(self) -> int:
        """计算LSTM输出维度"""
        multiplier = 2 if self.lstm_bidirectional else 1
        return self.lstm_hidden_dim * multiplier

    @property
    def classifier_input_dim(self) -> int:
        """计算分类器输入维度"""
        base_dim = self.lstm_output_dim
        if self.use_rule_features:
            return base_dim + self.rule_feature_dim
        return base_dim


# 默认配置
DEFAULT_CONFIG = ModelConfig(
    input_dims={'video': 20, 'audio': 15, 'text': 25},
    embedding_dim=128,
    transformer_layers=2,
    transformer_heads=4,
    transformer_ff_dim=512,
    transformer_dropout=0.1,
    lstm_hidden_dim=128,
    lstm_layers=2,
    lstm_dropout=0.3,
    lstm_bidirectional=True,
    attention_hidden_dim=64,
    num_classes=7,
    use_rule_features=True,
    rule_feature_dim=7,
    classifier_hidden_dim=128,
    classifier_dropout=0.3
)


# 轻量级配置（快速推理）
LIGHTWEIGHT_CONFIG = ModelConfig(
    input_dims={'video': 20, 'audio': 15, 'text': 25},
    embedding_dim=64,
    transformer_layers=1,
    transformer_heads=2,
    transformer_ff_dim=256,
    transformer_dropout=0.1,
    lstm_hidden_dim=64,
    lstm_layers=1,
    lstm_dropout=0.2,
    lstm_bidirectional=True,
    attention_hidden_dim=32,
    num_classes=7,
    use_rule_features=True,
    rule_feature_dim=7,
    classifier_hidden_dim=64,
    classifier_dropout=0.2
)


# 高精度配置（高性能）
HIGH_ACCURACY_CONFIG = ModelConfig(
    input_dims={'video': 20, 'audio': 15, 'text': 25},
    embedding_dim=256,
    transformer_layers=4,
    transformer_heads=8,
    transformer_ff_dim=1024,
    transformer_dropout=0.1,
    lstm_hidden_dim=256,
    lstm_layers=3,
    lstm_dropout=0.4,
    lstm_bidirectional=True,
    attention_hidden_dim=128,
    num_classes=7,
    use_rule_features=True,
    rule_feature_dim=7,
    classifier_hidden_dim=256,
    classifier_dropout=0.4
)
