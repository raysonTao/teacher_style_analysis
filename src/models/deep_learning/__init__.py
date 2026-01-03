"""深度学习模型模块

包含基于Transformer和LSTM的混合架构用于教师风格分析
"""

from .mman_model import MMANModel
from .config import ModelConfig

__all__ = ['MMANModel', 'ModelConfig']
