# 深度学习模型集成使用指南

## 概述

教师风格分析系统现已成功集成深度学习模型（MMAN架构），支持三种工作模式：

1. **规则系统模式** (`rule`) - 传统的规则驱动+机器学习混合方式
2. **深度学习模式** (`deep_learning`) - 纯深度学习模型推理
3. **混合模式** (`hybrid`) - 结合规则系统和深度学习的优势

## 快速开始

### 1. 基本使用（默认规则系统）

```python
from src.models.core.style_classifier import StyleClassifier

# 创建分类器（默认规则系统模式，向后兼容）
classifier = StyleClassifier()

# 执行分类
result = classifier.classify_style(features=your_features)
print(result['style_scores'])
```

### 2. 使用深度学习模式

```python
from src.models.core.style_classifier import StyleClassifier

# 创建深度学习分类器
classifier = StyleClassifier(
    mode='deep_learning',
    dl_checkpoint='./checkpoints/best_model.pth',
    dl_device='cuda'  # 或 'cpu'
)

# 执行分类
result = classifier.classify_style(features=your_features)

print(f"预测风格: {result.get('predicted_style')}")
print(f"置信度: {result.get('confidence'):.4f}")
print(f"Top-3风格: {result['top_styles']}")
```

### 3. 使用混合模式（推荐）

```python
from src.models.core.style_classifier import StyleClassifier

# 创建混合分类器
classifier = StyleClassifier(
    mode='hybrid',
    dl_checkpoint='./checkpoints/best_model.pth',
    dl_device='cuda'
)

# 执行分类
result = classifier.classify_style(features=your_features)

print(f"方法: {result.get('method')}")
print(f"混合得分: {result['style_scores']}")
print(f"深度学习结果: {result['deep_learning_results']}")
print(f"规则系统结果: {result['rule_based_results']}")
```

## API参考

### StyleClassifier 初始化参数

```python
StyleClassifier(
    mode: str = 'rule',                    # 工作模式：'rule', 'deep_learning', 'hybrid'
    dl_checkpoint: Optional[str] = None,   # 深度学习模型检查点路径
    dl_model_config: str = 'default',      # 模型配置：'default', 'lightweight', 'high_accuracy'
    dl_device: Optional[str] = None        # 设备：'cuda', 'cpu', None(自动选择)
)
```

### 分类结果结构

#### 规则系统模式
```python
{
    'style_scores': {              # 7种风格的得分
        '理论讲授型': 0.5,
        '启发引导型': 0.7,
        # ...
    },
    'top_styles': [                # Top-3 风格
        ('启发引导型', 0.7),
        ('理论讲授型', 0.5),
        # ...
    ],
    'rule_based_results': {...},   # 规则系统得分
    'ml_based_results': {...},     # ML层得分
    'feature_contributions': {...}, # 特征贡献度
    'confidence': 0.85,            # 置信度
    'method': 'rule'               # 使用的方法
}
```

#### 深度学习模式
```python
{
    'style_scores': {...},          # 7种风格的得分
    'predicted_style': '启发引导型', # 预测的主要风格
    'top_styles': [...],            # Top-3 风格
    'confidence': 0.75,             # 置信度
    'method': 'deep_learning'       # 使用的方法
}
```

#### 混合模式
```python
{
    'style_scores': {...},              # 混合后的得分（50/50权重）
    'top_styles': [...],                # Top-3 风格
    'deep_learning_results': {...},    # 深度学习模型结果
    'rule_based_results': {...},       # 规则系统结果
    'confidence': 0.80,                 # 平均置信度
    'method': 'hybrid'                  # 使用的方法
}
```

## 输入特征格式

分类器需要的特征格式：

```python
features = {
    'video': {
        'head_movement_frequency': float,       # 头部移动频率
        'body_movement_frequency': float,       # 身体移动频率
        'behavior_frequency': {
            'writing': float,                   # 书写行为频率
            'gesturing': float,                 # 手势频率
            'pointing': float,                  # 指点频率
            'standing': float,                  # 站立频率
            'walking': float                    # 走动频率
        },
        'eye_contact_score': float,             # 眼神交流得分
        'facial_expression_scores': {
            'neutral': float,
            'happy': float,
            'surprise': float,
            'sad': float,
            'angry': float,
            'disgust': float,
            'fear': float
        }
    },
    'audio': {
        'speech_rate': float,                   # 语速(字/分钟)
        'volume_level': float,                  # 音量水平
        'pitch_variation': float,               # 音调变化
        'silence_ratio': float,                 # 停顿比例
        'emotion_scores': {
            'neutral': float,
            'happy': float,
            'sad': float,
            'angry': float,
            'fear': float,
            'disgust': float,
            'surprise': float
        }
    },
    'text': {
        'vocabulary_richness': float,           # 词汇丰富度
        'sentence_complexity': float,           # 句子复杂度
        'question_frequency': float,            # 提问频率
        'keyword_density': {
            'definition': float,
            'example': float,
            'explanation': float,
            'summary': float,
            'question': float
        },
        'logical_indicators': {
            'causality': float,
            'comparison': float,
            'sequence': float,
            'emphasis': float
        },
        'sentiment_score': float                # 情感得分
    },
    'fusion': {                                 # 融合特征（用于规则系统和混合模式）
        'interaction_level': float,
        'explanation_clarity': float,
        'emotional_engagement': float,
        'logical_structure': float,
        'teaching_style_metrics': {             # 预计算的风格指标
            'lecturing': float,
            'guiding': float,
            'interactive': float,
            'logical': float,
            'problem_driven': float,
            'emotional': float,
            'patient': float
        }
    }
}
```

## 模型配置

### Default（默认配置）
- 参数量: ~1.1M
- 训练速度: 中等
- 准确率: 高
- 推荐场景: 生产环境

### Lightweight（轻量级配置）
- 参数量: ~300K
- 训练速度: 快
- 准确率: 中等
- 推荐场景: 快速原型、实时推理、资源受限环境

### High Accuracy（高精度配置）
- 参数量: ~4M
- 训练速度: 慢
- 准确率: 最高
- 推荐场景: 离线分析、追求最佳性能

## 性能对比

当前训练好的模型性能（基于合成数据）：

| 指标 | 值 |
|------|-----|
| 准确率 | 0.4533 |
| F1 (macro) | 0.4224 |
| F1 (weighted) | 0.4489 |
| AUC (OvR) | 0.8078 |
| AUC (OvO) | 0.8018 |

**注意**: 这是基于合成数据训练的结果。使用真实MM-TBA数据集训练后，性能预计会显著提升。

## 模式选择建议

### 何时使用规则系统模式？
- ✅ 需要完全可解释的结果
- ✅ 数据量较小或质量不确定
- ✅ 向后兼容现有系统
- ✅ 不依赖GPU

### 何时使用深度学习模式？
- ✅ 有充足的高质量训练数据
- ✅ 追求更高的准确率
- ✅ 数据分布复杂，规则难以覆盖
- ✅ 可以接受黑盒模型

### 何时使用混合模式？（推荐）
- ✅ 兼顾准确率和可解释性
- ✅ 利用规则系统的领域知识
- ✅ 深度学习模型提供额外信号
- ✅ 提高鲁棒性（深度学习失败时回退到规则）
- ✅ **最稳定可靠的选择**

## 故障排查

### GPU相关问题

#### cuDNN版本不兼容
```bash
# 设置环境变量指向正确的CUDA版本
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.7
```

或者使用CPU模式：
```python
classifier = StyleClassifier(mode='deep_learning', dl_device='cpu')
```

#### GPU内存不足
- 使用CPU模式
- 使用lightweight配置
- 减小batch_size（如果批量处理）

### 模型加载失败

如果深度学习模型加载失败，系统会自动回退到规则系统：
```
WARNING - 深度学习模型加载失败，将使用规则系统
WARNING - 回退到规则系统模式
```

确保检查点文件存在：
```bash
ls -lh checkpoints/best_model.pth
```

### 特征维度问题

确保输入特征包含所有必要字段。如果某些字段缺失，会自动填充0值。

## 下一步计划

1. **使用真实MM-TBA数据集训练** - 预计准确率提升到70%+
2. **优化推理速度** - 针对生产环境优化
3. **添加模型解释性** - 集成Grad-CAM等可视化工具
4. **支持在线学习** - 持续学习新的教学风格

## 示例代码

完整示例见 `test_integration.py`。

## 联系与支持

如有问题，请查看：
- 深度学习模型文档: `src/models/deep_learning/README.md`
- 训练脚本: `src/models/deep_learning/train.py`
- 推理包装器: `src/models/deep_learning/inference.py`
