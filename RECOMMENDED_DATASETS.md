# 教学风格分类数据集推荐

## 🎯 强烈推荐的数据集

### 1. **T-MED (Teacher Multimodal Emotion Dataset)** ⭐⭐⭐⭐⭐

**最新！2024年12月发布，最适合教学风格分类！**

#### 📊 数据集概况
- **规模**: 14,938 个样本
- **教师数量**: 200+ 教师
- **总时长**: 17+ 小时视频
- **来源**: 250 个真实课堂（K-12 到高等教育）
- **学科**: 11 个学科

#### 🏷️ 标签信息（8种情感标签）
```
✅ 与教学风格高度相关：
1. Neutral (中性)
2. Anger (愤怒)
3. Joy (愉悦)
4. Surprise (惊讶)
5. Sadness (悲伤)
6. Patience (耐心) ← 教师特有
7. Enthusiasm (热情) ← 教师特有
8. Expectation (期待) ← 教师特有
```

**为什么最适合？**
- ✅ 专门针对教师情感/风格设计
- ✅ 包含"耐心"、"热情"等教学风格相关标签
- ✅ 多模态完整：视频、音频、文本、教学信息
- ✅ 数据量适中，质量高
- ✅ 真实课堂数据

#### 🔗 获取方式
- **论文**: [arXiv:2512.20548](https://arxiv.org/abs/2512.20548)
- **下载**: 需要联系作者或查看论文中的数据获取说明
- **发布时间**: 2024年12月

#### 💡 使用建议
```python
# 标签映射到我们的7种教学风格
T-MED 标签 → 我们的风格标签
Enthusiasm + Joy → 情感表达型
Patience → 耐心细致型
Neutral + structured content → 理论讲授型
Enthusiasm + interaction → 启发引导型
```

---

### 2. **TBU (Teacher Behavior Understanding Dataset)** ⭐⭐⭐⭐

**大规模，多任务，GitHub开源！**

#### 📊 数据集概况
- **规模**: 37,026 高质量教学行为片段
- **标注片段**: 9,422 个（带时间边界）
- **描述片段**: 6,098 个（多层次标注）
- **教育阶段**: 4个（小学、初中、高中、大学）

#### 🏷️ 标注类型
```
1. 细粒度行为标签
2. 空间位置标注
3. 交互对象标注
4. 时间边界标注
```

#### ✅ 支持的任务
- 行为识别 (Behavior Recognition)
- 行为检测 (Behavior Detection)
- 行为描述 (Behavior Description)

#### 🔗 获取方式
- **GitHub**: [https://github.com/cai-KU/TBU](https://github.com/cai-KU/TBU)
- **论文**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1077314225000992)
- **状态**: ✅ 公开可下载

#### 💡 使用建议
- 可用于预训练
- 提取教师行为特征
- 结合行为模式推断教学风格

---

### 3. **SCB-Dataset (Student and Teacher Classroom Behavior)** ⭐⭐⭐⭐

**开源，图像+标注，GitHub可用！**

#### 📊 数据集概况
- **类别**: 19 个行为类别
- **目标检测部分**: 13,330 图像，122,977 标签
- **图像分类部分**: 21,019 图像
- **包含**: 学生和教师的课堂行为

#### 🏷️ 行为类别（部分）
```
教师行为：
- 讲解 (Explaining)
- 写板书 (Writing)
- 指点 (Pointing)
- 互动 (Interacting)
- 走动 (Walking)
等...
```

#### 🔗 获取方式
- **GitHub**: [https://github.com/Whiffe/SCB-dataset](https://github.com/Whiffe/SCB-dataset)
- **论文**: [arXiv:2304.02488](https://arxiv.org/abs/2304.02488)
- **状态**: ✅ 公开可下载

#### 💡 使用建议
- 用于视觉特征提取
- 行为检测预训练
- 与其他数据集结合使用

---

### 4. **MUTLA (Multimodal Teaching and Learning Analytics)** ⭐⭐⭐

**大规模，学习分析，多模态完整！**

#### 📊 数据集概况
- **规模**: 大规模学习记录
- **模态**: 视频、EEG脑电波、学习日志
- **场景**: Squirrel AI Learning System 的学生学习过程
- **时长**: 约2小时/会话

#### 🔗 获取方式
- **GitHub**: [https://github.com/RyanH98/SAILData](https://github.com/RyanH98/SAILData)
- **短链接**: [https://tinyurl.com/SAILdata](https://tinyurl.com/SAILdata)
- **论文**: [arXiv:1910.06078](https://arxiv.org/abs/1910.06078)
- **状态**: ✅ 公开可下载

#### 💡 使用建议
- 更偏向学生学习分析
- 可作为补充数据集
- EEG数据较特殊，可能需要额外设备

---

## 📈 数据集对比

| 数据集 | 规模 | 标签类型 | 下载难度 | 适合度 | 推荐指数 |
|--------|------|----------|----------|--------|----------|
| **T-MED** | 14,938 | 8种情感（含教学风格） | ⚠️ 需联系作者 | ⭐⭐⭐⭐⭐ | **首选** |
| **TBU** | 37,026 | 行为+位置+对象 | ✅ GitHub直接下载 | ⭐⭐⭐⭐ | 推荐 |
| **SCB-Dataset** | 13,330图像 | 19种行为 | ✅ GitHub直接下载 | ⭐⭐⭐ | 推荐 |
| **MUTLA** | 大规模 | 学习分析 | ✅ GitHub直接下载 | ⭐⭐⭐ | 辅助 |
| **MM-TBA** | 209 | ❌ 无风格标签 | ✅ 已有 | ❌ | 不推荐 |

---

## 🎯 推荐使用策略

### 策略 1: T-MED 为主（最佳）

```bash
# 1. 获取 T-MED 数据集
#    访问 https://arxiv.org/abs/2512.20548
#    联系作者获取数据

# 2. 标签映射
#    将8种情感映射到7种教学风格
#    Enthusiasm + Patience → 不同的教学风格

# 3. 训练模型
python -m src.models.deep_learning.train \
    --data_path data/t-med/processed.json \
    --batch_size 64 \
    --num_epochs 150 \
    --device cuda
```

**预期效果：**
- 准确率: 65-75%
- 标签质量: ⭐⭐⭐⭐⭐
- 泛化能力: 强

---

### 策略 2: TBU + SCB 组合（开源）

```bash
# 1. 下载数据集
git clone https://github.com/cai-KU/TBU
git clone https://github.com/Whiffe/SCB-dataset

# 2. 提取特征
#    从行为标注推断教学风格
#    Writing + Explaining → 理论讲授型
#    Pointing + Interacting → 互动导向型

# 3. 合并训练
#    使用TBU的行为特征
#    结合SCB的视觉特征
```

**预期效果：**
- 准确率: 55-65%
- 标签质量: ⭐⭐⭐⭐
- 优势: 完全开源

---

### 策略 3: 多数据集联合训练（高级）

```
Step 1: TBU 预训练（行为识别）
  ↓
Step 2: T-MED 微调（情感/风格）
  ↓
Step 3: SCB-Dataset 增强（视觉特征）
  ↓
Result: 高性能教学风格分类模型
```

**预期效果：**
- 准确率: 70-80%
- 标签质量: ⭐⭐⭐⭐⭐
- 训练时间: 2-4小时

---

## 📝 数据集获取步骤

### T-MED（推荐）

1. **访问论文**
   ```
   https://arxiv.org/abs/2512.20548
   ```

2. **查找数据获取信息**
   - 查看论文的 "Dataset Availability" 部分
   - 通常会提供下载链接或联系方式

3. **联系作者**（如果需要）
   ```
   通过论文中的email联系第一作者
   说明研究用途和目的
   ```

4. **数据处理**
   - 下载后需要转换为我们的格式
   - 标签映射（8种情感 → 7种风格）

---

### TBU（最简单）

```bash
# 直接克隆
git clone https://github.com/cai-KU/TBU

# 查看数据结构
cd TBU
ls -lh

# 阅读README了解格式
cat README.md
```

---

### SCB-Dataset（最简单）

```bash
# 直接克隆
git clone https://github.com/Whiffe/SCB-dataset

# 下载可能需要时间（图像数据较大）
cd SCB-dataset
```

---

## 🔧 数据转换工具

### 创建通用转换脚本

```python
# convert_dataset.py
"""
通用数据集转换脚本
支持: T-MED, TBU, SCB-Dataset
"""

def convert_tmed_to_format(tmed_data):
    """转换 T-MED 数据"""
    # 情感标签映射
    emotion_to_style = {
        'Enthusiasm': ['情感表达型', '启发引导型'],
        'Patience': ['耐心细致型'],
        'Joy': ['情感表达型'],
        'Neutral': ['理论讲授型', '逻辑推导型'],
        # ...
    }
    pass

def convert_tbu_to_format(tbu_data):
    """转换 TBU 数据"""
    # 行为映射
    behavior_to_style = {
        'Writing': '理论讲授型',
        'Explaining': '理论讲授型',
        'Pointing': '启发引导型',
        'Interacting': '互动导向型',
        # ...
    }
    pass

def convert_scb_to_format(scb_data):
    """转换 SCB-Dataset 数据"""
    pass
```

---

## 🎯 立即行动方案

### 方案 A: 最快可用（今天可完成）

```bash
# 1. 下载 TBU 或 SCB-Dataset（GitHub直接下载）
git clone https://github.com/cai-KU/TBU

# 2. 编写简单的转换脚本

# 3. 开始训练
```

**优势**: 立即可用，完全开源
**劣势**: 需要手动映射行为到风格

---

### 方案 B: 最佳效果（1-2天可完成）

```bash
# 1. 联系 T-MED 作者获取数据
#    Email: 查看 https://arxiv.org/abs/2512.20548

# 2. 等待作者回复（通常1-2天）

# 3. 下载并转换数据

# 4. 训练高质量模型
```

**优势**: 最佳标签质量，效果最好
**劣势**: 需要等待作者回复

---

### 方案 C: 混合策略（推荐）

```bash
# 第1步（今天）: 使用 TBU 开始实验
git clone https://github.com/cai-KU/TBU
# 快速训练一个baseline

# 第2步（同时进行）: 申请 T-MED 数据
# 发送email给T-MED作者

# 第3步（收到数据后）: 用 T-MED 重新训练
# 获得最佳效果
```

---

## 📚 相关资源

### 论文列表

1. **T-MED**: [Advancing Multimodal Teacher Sentiment Analysis](https://arxiv.org/abs/2512.20548)
2. **TBU**: [Classroom teacher behavior analysis: The TBU dataset](https://www.sciencedirect.com/science/article/abs/pii/S1077314225000992)
3. **SCB-Dataset**: [A Dataset for Detecting Student and Teacher Classroom Behavior](https://arxiv.org/abs/2304.02488)
4. **MUTLA**: [A Large-Scale Dataset for Multimodal Teaching and Learning Analytics](https://arxiv.org/abs/1910.06078)

### GitHub 仓库

- TBU: https://github.com/cai-KU/TBU
- SCB-Dataset: https://github.com/Whiffe/SCB-dataset
- MUTLA: https://github.com/RyanH98/SAILData

---

## 🤝 需要帮助？

我可以帮您：
1. ✅ 联系数据集作者（起草email）
2. ✅ 编写数据转换脚本
3. ✅ 设计训练策略
4. ✅ 调优模型参数

---

## 📊 总结

| 优先级 | 数据集 | 行动 | 预期时间 |
|--------|--------|------|----------|
| **P0** | T-MED | 联系作者获取 | 1-3天 |
| **P1** | TBU | 立即下载使用 | 今天 |
| **P2** | SCB-Dataset | 辅助使用 | 今天 |
| **P3** | MUTLA | 可选补充 | 按需 |

**立即开始：**
1. 下载 TBU 数据集（5分钟）
2. 同时发email给 T-MED 作者
3. 今天就能开始训练！
