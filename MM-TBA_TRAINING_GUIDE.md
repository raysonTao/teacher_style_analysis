# MM-TBA 数据集训练指南

## 数据集处理完成 ✅

您的 MM-TBA 数据集已成功解压到：
```
/home/rayson/code/teacher_style_analysis/data/mm-tba/MM-TBA/
```

数据集包含：
- **354 个教师样本** (metadata.xlsx)
- **167 个训练样本** (Teacher_Lecture_Evaluation/finetune_data/train.json)
- **42 个评估样本** (Teacher_Lecture_Evaluation/finetune_data/eval.json)
- 教师行为检测数据 (Teacher_Action_Detection/)
- 教学设计数据 (Teacher_Instructional_Design/)

## 训练方案

### 方案 1: 使用合成数据快速训练（推荐先执行）

**优势：** 快速验证 GPU 训练流程，熟悉训练参数

```bash
cd /home/rayson/code/teacher_style_analysis

# 直接运行GPU训练脚本
./train_gpu.sh
```

**或者手动执行：**

```bash
# 设置CUDA环境变量
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.7

# 使用合成数据训练（5000样本）
python -m src.models.deep_learning.train \
    --use_synthetic \
    --num_synthetic 5000 \
    --model_config default \
    --batch_size 64 \
    --num_epochs 200 \
    --lr 1e-4 \
    --device cuda \
    --num_workers 4 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

**预期结果：**
- 训练时间：约 30-60 分钟（GPU）
- 准确率：45-50%（合成数据基线）
- 模型保存：`./checkpoints/best_model.pth`

---

### 方案 2: 使用 MM-TBA 真实数据训练（推荐）

**优势：** 真实教学数据，预期更高准确率

#### 步骤 1: 转换数据格式

```bash
cd /home/rayson/code/teacher_style_analysis

# 运行数据转换脚本
python convert_mmtba.py
```

这会将 MM-TBA 的讲课文本转换为我们需要的特征格式：
- 输入：167 训练样本 + 42 评估样本 = 209 样本
- 输出：`data/mm-tba/mmtba_converted.json`
- 格式：包含 video_features (20维), audio_features (15维), text_features (25维)

#### 步骤 2: GPU 训练

**选项 A: 一键训练（推荐）**

```bash
# 自动转换数据并训练
./train_mmtba_gpu.sh
```

**选项 B: 手动训练（更多控制）**

```bash
# 设置CUDA环境变量
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.7

# 基础配置训练
python -m src.models.deep_learning.train \
    --data_path data/mm-tba/mmtba_converted.json \
    --model_config default \
    --batch_size 32 \
    --num_epochs 150 \
    --lr 5e-5 \
    --device cuda \
    --checkpoint_dir ./checkpoints/mmtba \
    --log_dir ./logs/mmtba

# 高精度配置训练（更好的效果但更慢）
python -m src.models.deep_learning.train \
    --data_path data/mm-tba/mmtba_converted.json \
    --model_config high_accuracy \
    --batch_size 16 \
    --num_epochs 200 \
    --lr 3e-5 \
    --device cuda \
    --checkpoint_dir ./checkpoints/mmtba_high \
    --log_dir ./logs/mmtba_high

# 轻量级配置训练（更快但准确率稍低）
python -m src.models.deep_learning.train \
    --data_path data/mm-tba/mmtba_converted.json \
    --model_config lightweight \
    --batch_size 64 \
    --num_epochs 100 \
    --lr 1e-4 \
    --device cuda \
    --checkpoint_dir ./checkpoints/mmtba_light \
    --log_dir ./logs/mmtba_light
```

**预期结果：**
- 训练时间：约 15-30 分钟（GPU）
- 准确率：55-70%（真实数据）
- 模型保存：`./checkpoints/mmtba/best_model.pth`

---

### 方案 3: 先合成数据预训练，再用 MM-TBA 微调（最佳效果）

```bash
# 步骤1: 合成数据预训练
python -m src.models.deep_learning.train \
    --use_synthetic \
    --num_synthetic 5000 \
    --batch_size 64 \
    --num_epochs 100 \
    --device cuda \
    --checkpoint_dir ./checkpoints/pretrain

# 步骤2: 转换MM-TBA数据
python convert_mmtba.py

# 步骤3: 用MM-TBA数据微调
python -m src.models.deep_learning.train \
    --data_path data/mm-tba/mmtba_converted.json \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-5 \
    --device cuda \
    --resume ./checkpoints/pretrain/best_model.pth \
    --checkpoint_dir ./checkpoints/finetuned \
    --log_dir ./logs/finetuned
```

**预期结果：**
- 训练时间：约 45-90 分钟（GPU）
- 准确率：60-75%（预训练+微调）
- 模型保存：`./checkpoints/finetuned/best_model.pth`

---

## 训练参数说明

### 模型配置

| 配置 | 参数量 | 训练速度 | 准确率 | 推荐场景 |
|------|--------|----------|--------|----------|
| `lightweight` | ~300K | 快 | 中等 | 快速实验、资源受限 |
| `default` | ~1.1M | 中等 | 高 | **推荐：生产环境** |
| `high_accuracy` | ~4M | 慢 | 最高 | 追求最佳性能 |

### 关键参数调优

```bash
# Batch Size（根据GPU显存调整）
--batch_size 64   # 建议值：16-64，显存大用大值
--batch_size 32   # 小数据集推荐32

# 学习率
--lr 1e-4         # 合成数据、预训练
--lr 5e-5         # 真实数据训练
--lr 1e-5         # 微调已有模型

# 训练轮数
--num_epochs 100  # 轻量级模型
--num_epochs 150  # 默认配置
--num_epochs 200  # 高精度配置

# 早停耐心值
--early_stopping 10   # 快速实验
--early_stopping 20   # 标准训练

# 优化器
--optimizer adamw     # 推荐（效果最好）
--optimizer adam      # 备选
--optimizer sgd       # 经典方法

# 学习率调度器
--scheduler cosine    # 推荐（平滑衰减）
--scheduler step      # 阶梯衰减
--scheduler plateau   # 自适应
```

---

## 训练监控

### 查看训练日志

```bash
# 实时查看训练进度
tail -f logs/train.log

# 查看TensorBoard（如果安装）
tensorboard --logdir logs/
```

### 检查模型文件

```bash
# 查看保存的检查点
ls -lh checkpoints/

# 查看最佳模型
ls -lh checkpoints/best_model.pth
```

---

## 使用训练好的模型

### 方法 1: 命令行使用

```bash
# 分析单个视频（使用训练好的模型）
python -m src.main analyze \
    --video data/videos/lesson.mp4 \
    --teacher teacher001 \
    --discipline "数学" \
    --grade "高中" \
    --mode deep_learning \
    --device cuda

# 如果使用MM-TBA训练的模型，需要修改main.py中的检查点路径
# 或者手动指定检查点（需要修改代码）
```

### 方法 2: Python 脚本使用

```python
from src.models.core.style_classifier import StyleClassifier

# 创建分类器（使用MM-TBA训练的模型）
classifier = StyleClassifier(
    mode='deep_learning',
    dl_checkpoint='./checkpoints/mmtba/best_model.pth',  # MM-TBA模型
    dl_model_config='default',
    dl_device='cuda'
)

# 执行分类
result = classifier.classify_style(features=your_features)

print(f"预测风格: {result['predicted_style']}")
print(f"置信度: {result['confidence']:.4f}")
print(f"Top-3: {result['top_styles']}")
```

---

## 常见问题

### Q1: GPU 内存不足怎么办？

```bash
# 减小batch size
--batch_size 16  # 或更小

# 使用轻量级配置
--model_config lightweight

# 减少worker数量
--num_workers 2
```

### Q2: cuDNN 版本不兼容？

```bash
# 方案1: 设置CUDA环境变量
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.7

# 方案2: 使用CPU训练
--device cpu
```

### Q3: 训练速度太慢？

```bash
# 增加worker数量
--num_workers 8

# 使用更大batch size
--batch_size 128

# 使用轻量级模型
--model_config lightweight

# 减少训练轮数
--num_epochs 50
```

### Q4: 准确率不高怎么办？

1. **增加训练数据：** 使用方案3（预训练+微调）
2. **调整学习率：** 尝试不同学习率（1e-5 到 1e-3）
3. **增加训练轮数：** `--num_epochs 200`
4. **使用高精度配置：** `--model_config high_accuracy`
5. **数据增强：** 在转换脚本中添加噪声

---

## 快速开始（推荐流程）

```bash
# 1. 进入项目目录
cd /home/rayson/code/teacher_style_analysis

# 2. 运行MM-TBA数据训练（一键完成）
./train_mmtba_gpu.sh

# 3. 等待训练完成（约15-30分钟）

# 4. 测试模型
python test_integration.py
```

**就这么简单！🎉**

---

## 预期训练结果

### 合成数据基线
- Accuracy: ~45%
- F1 (macro): ~42%
- AUC: ~80%

### MM-TBA 真实数据
- Accuracy: **55-70%** （预期）
- F1 (macro): **50-65%** （预期）
- AUC: **85-90%** （预期）

### 预训练+微调
- Accuracy: **60-75%** （预期）
- F1 (macro): **55-70%** （预期）
- AUC: **88-93%** （预期）

---

## 下一步

1. ✅ 数据已解压和处理
2. 🚀 **立即开始：** 运行 `./train_mmtba_gpu.sh`
3. 📊 训练完成后查看结果
4. 🎯 使用训练好的模型分析视频
5. 🔧 根据结果调优参数

需要帮助？查看详细文档：
- 深度学习集成：`DEEP_LEARNING_INTEGRATION.md`
- 训练脚本文档：`src/models/deep_learning/README.md`
