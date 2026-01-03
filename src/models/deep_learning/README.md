# MMAN深度学习模型 - 使用说明

## 模型架构

**MMAN (Multi-Modal Attention Network)** - 基于Transformer和LSTM的混合架构

### 核心特点
- ✅ 多模态特征融合（视频、音频、文本）
- ✅ Transformer跨模态注意力机制
- ✅ LSTM时序建模
- ✅ 规则系统集成
- ✅ 7种教学风格分类

### 模型统计
- **参数量**: ~1.1M (默认配置)
- **输入**: 视频(20维) + 音频(15维) + 文本(25维)
- **输出**: 7种教学风格概率分布

## 快速开始

### 1. 训练模型（使用合成数据测试）

```bash
cd /home/rayson/code/teacher_style_analysis

# 使用默认配置训练
python3 src/models/deep_learning/train.py \
    --use_synthetic \
    --num_synthetic 500 \
    --batch_size 16 \
    --num_epochs 20 \
    --lr 1e-4 \
    --device cuda

# 使用轻量级配置（快速测试）
python3 src/models/deep_learning/train.py \
    --use_synthetic \
    --model_config lightweight \
    --batch_size 32 \
    --num_epochs 10 \
    --device cuda
```

### 2. 训练模型（使用真实数据）

```bash
# ��要先准备数据集（JSON格式）
python3 src/models/deep_learning/train.py \
    --data_path /mnt/data/teacher_style_analysis/MM-TBA/processed_data.json \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --early_stopping 15 \
    --device cuda
```

### 3. 测试模型

```bash
python3 src/models/deep_learning/train.py \
    --test_only \
    --test_checkpoint ./checkpoints/best_model.pth \
    --data_path /path/to/test/data.json \
    --device cuda
```

### 4. 从检查点恢复训练

```bash
python3 src/models/deep_learning/train.py \
    --resume ./checkpoints/best_model.pth \
    --num_epochs 50 \
    --device cuda
```

## 命令行参数说明

### 数据参数
- `--data_path`: 数据路径（JSON文件或目录）
- `--use_synthetic`: 使用合成数据（测试用）
- `--num_synthetic`: 合成数据样本数（默认1000）

### 模型参数
- `--model_config`: 模型配置
  - `default`: 平衡型（推荐）
  - `lightweight`: 轻量级（快速）
  - `high_accuracy`: 高精度（大模型）
- `--use_rule_features`: 使用规则系统特征（默认True）

### 训练参数
- `--batch_size`: 批次大小（默认32）
- `--num_epochs`: 训练轮数（默认100）
- `--lr`: 学习率（默认1e-4）
- `--weight_decay`: 权重衰减（默认1e-5）
- `--optimizer`: ��化器（adam/adamw/sgd）
- `--scheduler`: 学习率调度器（none/step/cosine/plateau）
- `--early_stopping`: 早停耐心值（默认10）

### 设备参数
- `--device`: 训练设备（cuda/cpu）
- `--seed`: 随机种子（默认42）

## 数据格式

训练数据需要是JSON格式，每个样本包含：

```json
{
  "sample_id": "sample_0001",
  "split": "train",  // 或 "val", "test"
  "label": 0,  // 0-6的整数，对应7种风格
  "video_features": [20维浮点数数组],
  "audio_features": [15维浮点数数组],
  "text_features": [25维浮点数数组],
  "rule_features": [7维浮点数数组]  // 可选
}
```

## 7种教学风格标签

0. 理论讲授型
1. 启发引导型
2. 互动导向型
3. 逻辑推导型
4. 题目驱动型
5. 情感表达型
6. 耐心细致型

## 输出文件

训练过程会生成以下文件：

```
checkpoints/
├── best_model.pth           # 最佳模型检查点
├── best_metrics.json        # 最佳模型的评估指标
└── checkpoint_epoch_*.pth   # 定期保存的检查点

logs/
├── training_curves.png      # 训练曲线
├── confusion_matrix.png     # 混淆矩阵
├── per_class_metrics.png    # 每类指标柱状图
└── test_metrics.json        # 测试指标
```

## Python API使用示例

```python
import torch
from src.models.deep_learning.mman_model import create_model

# 创建模型
model = create_model('default')
model.eval()

# 准备输入
features = {
    'video': torch.randn(1, 20),
    'audio': torch.randn(1, 15),
    'text': torch.randn(1, 25)
}
rule_features = torch.randn(1, 7)

# 推理
with torch.no_grad():
    output = model(features, rule_features)
    predictions = output['predictions']
    probabilities = output['probabilities']

print(f"预测类别: {predictions}")
print(f"概率分布: {probabilities}")
```

## 性能对比

| 配置 | 参数量 | 训练速度 | 准确率 | 推荐场景 |
|------|--------|----------|--------|----------|
| lightweight | ~300K | 快 | 中 | 快速原型、实时推理 |
| default | ~1.1M | 中 | 高 | 生产环境（推荐）|
| high_accuracy | ~4M | 慢 | 最高 | 离线分析、最佳性能 |

## 依赖要求

```bash
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.60.0
```

## 故障排查

### GPU内存不足
```bash
# 减小batch size
--batch_size 8

# 使用轻量级配置
--model_config lightweight
```

### 训练不收敛
```bash
# 降低学习率
--lr 5e-5

# 更换优化器
--optimizer adamw

# 增加早停耐心值
--early_stopping 20
```

### 过拟合
```bash
# 增加权重衰减
--weight_decay 1e-4

# 减小模型
--model_config lightweight
```

## 联系与支持

如有问题，请查看日志文件或联系开发团队。
