# TBU 数据集 VLM 标注系统

使用公司内部 Claude API 对 TBU 数据集进行全自动教学风格标注。

## 系统架构

```
TBU 原始数据
    ↓
convert_tbu.py (转换为标注格式)
    ↓
vlm_annotator.py (VLM 批量标注)
    ↓
annotate_tbu.py (转换为训练格式)
    ↓
train.py (训练深度学习模型)
    ↓
训练好的模型
```

## 快速开始

### 步骤 0: 安装依赖

```bash
pip install anthropic tqdm
```

### 步骤 1: 下载 TBU 数据集

```bash
# 克隆 TBU 数据集
git clone https://github.com/cai-KU/TBU data/TBU

# 或者使用已有的 TBU 数据路径
```

### 步骤 2: 测试 VLM 标注器

```bash
# 测试 API 是否正常工作
chmod +x test_vlm_annotator.sh
./test_vlm_annotator.sh
```

**预期输出**:
```
VLM标注器快速测试
==================================
配置信息:
  API地址: https://aidev.deyecloud.com/api
  ...

开始测试标注器...

标注结果:
预测风格: 理论讲授型
置信度: 0.85
理由: 教师以讲解和板书为主，注重知识传授...

测试成功！
```

### 步骤 3: 小规模测试（推荐）

```bash
# 先标注 10 个样本测试流程
chmod +x annotate_and_train_tbu.sh
./annotate_and_train_tbu.sh data/TBU 10
```

**说明**:
- 第一个参数: TBU 数据路径
- 第二个参数: 最大样本数（可选，用于测试）

**预期时间**: 约 5-10 分钟（10个样本）

### 步骤 4: 全量标注和训练

```bash
# 标注全部数据（可能需要数小时）
./annotate_and_train_tbu.sh data/TBU
```

**预期时间**:
- TBU 数据集: ~37,000 个样本
- 标注速度: ~5-10 个/分钟
- 总时间: **3-12 小时**

**断点续传**:
如果中途中断，可以使用 `--resume` 参数继续：

```bash
python annotate_tbu.py annotate \
    --input data/tbu/tbu_for_annotation.json \
    --output data/tbu/tbu_annotated.json \
    --resume 1000  # 从第 1000 个样本继续
```

---

## 详细使用说明

### 1. 转换 TBU 数据

```bash
python -m src.annotation.convert_tbu \
    --tbu_path data/TBU \
    --output data/tbu/tbu_for_annotation.json \
    --max_samples 100  # 可选：限制样本数
```

**输出**: `data/tbu/tbu_for_annotation.json`

**格式**:
```json
[
  {
    "sample_id": "tbu_00001",
    "behavior_sequence": ["Writing", "Explaining", "Pointing"],
    "behavior_durations": {
      "Writing": 120.5,
      "Explaining": 280.3,
      "Pointing": 45.2
    },
    "video_frames": [
      "path/to/frame1.jpg",
      "path/to/frame2.jpg"
    ],
    "lecture_text": "今天我们学习...",
    "metadata": {
      "discipline": "数学",
      "grade": "初中"
    }
  }
]
```

### 2. VLM 批量标注

```bash
python annotate_tbu.py annotate \
    --input data/tbu/tbu_for_annotation.json \
    --output data/tbu/tbu_annotated.json \
    --model claude-3-5-sonnet-20241022 \
    --save_interval 10 \
    --max_samples 100  # 可选：测试用
```

**参数说明**:
- `--input`: 转换后的 TBU 数据
- `--output`: 标注结果输出路径
- `--model`: 使用的模型（默认: claude-3-5-sonnet-20241022）
- `--save_interval`: 每标注 N 个样本保存一次（防止中断丢失）
- `--resume`: 断点续传，从第 N 个样本继续
- `--max_samples`: 限制标注样本数（测试用）

**输出**: `data/tbu/tbu_annotated.json`

**格式**:
```json
[
  {
    "sample_id": "tbu_00001",
    "annotation": {
      "style": "理论讲授型",
      "confidence": 0.85,
      "reasoning": "教师以讲解和板书为主...",
      "secondary_style": "逻辑推导型",
      "key_features": ["板书频繁", "讲解系统", "逻辑清晰"]
    },
    "source_data": { ... }
  }
]
```

### 3. 转换为训练格式

```bash
python annotate_tbu.py convert \
    --input data/tbu/tbu_annotated.json \
    --output data/tbu/tbu_training.json \
    --train_ratio 0.7 \
    --val_ratio 0.15
```

**参数说明**:
- `--train_ratio`: 训练集比例（默认: 0.7）
- `--val_ratio`: 验证集比例（默认: 0.15）
- 测试集比例 = 1 - train_ratio - val_ratio

**输出**: `data/tbu/tbu_training.json`

### 4. 训练模型

```bash
python -m src.models.deep_learning.train \
    --data_path data/tbu/tbu_training.json \
    --model_config default \
    --batch_size 64 \
    --num_epochs 150 \
    --lr 5e-5 \
    --device cuda \
    --checkpoint_dir ./checkpoints/tbu
```

**预期结果**:
- 训练时间: 1-2 小时（37,000 样本，GPU）
- 准确率: 65-75%
- 模型文件: `./checkpoints/tbu/best_model.pth`

---

## API 配置

### 环境变量

```bash
# 公司内部 Claude API
export ANTHROPIC_BASE_URL="https://aidev.deyecloud.com/api"
export ANTHROPIC_AUTH_TOKEN="cr_fd8489bac5fac5a8cc9d234e8a93baf15c65a0fa96e64731c3f36201fe0417b1"
```

### Python 代码中使用

```python
from src.annotation.vlm_annotator import VLMStyleAnnotator

# 创建标注器
annotator = VLMStyleAnnotator(
    api_key="your-api-token",
    base_url="https://aidev.deyecloud.com/api",
    model="claude-3-5-sonnet-20241022"
)

# 标注单个样本
result = annotator.annotate_single_sample(
    behavior_sequence=['Writing', 'Explaining'],
    behavior_durations={'Writing': 120, 'Explaining': 280},
    lecture_text="今天我们学习..."
)

print(f"预测风格: {result['style']}")
print(f"置信度: {result['confidence']}")
```

---

## 标注质量控制

### 1. 查看标注统计

```python
import json

with open('data/tbu/tbu_annotated.json', 'r') as f:
    data = json.load(f)

# 统计置信度
confidences = [s['annotation']['confidence'] for s in data]
print(f"平均置信度: {sum(confidences)/len(confidences):.3f}")
print(f"高置信度 (>0.8): {sum(1 for c in confidences if c > 0.8)}")
print(f"低置信度 (<0.5): {sum(1 for c in confidences if c < 0.5)}")

# 统计风格分布
from collections import Counter
styles = [s['annotation']['style'] for s in data]
print(Counter(styles))
```

### 2. 人工审核低置信度样本

```bash
# 筛选低置信度样本
python -c "
import json
with open('data/tbu/tbu_annotated.json', 'r') as f:
    data = json.load(f)

low_conf = [s for s in data if s['annotation']['confidence'] < 0.5]
print(f'低置信度样本: {len(low_conf)}')

with open('data/tbu/low_confidence_samples.json', 'w') as f:
    json.dump(low_conf, f, ensure_ascii=False, indent=2)
"
```

### 3. 重新标注特定样本

```python
from src.annotation.vlm_annotator import VLMStyleAnnotator
import json

# 加载低置信度样本
with open('data/tbu/low_confidence_samples.json', 'r') as f:
    low_conf_samples = json.load(f)

# 使用不同温度重新标注
annotator = VLMStyleAnnotator(
    api_key="...",
    base_url="...",
    model="claude-3-5-sonnet-20241022"
)

# 重新标注
for sample in low_conf_samples[:10]:  # 前10个
    result = annotator.annotate_single_sample(**sample['source_data'])
    print(f"{sample['sample_id']}: {result['style']} ({result['confidence']:.2f})")
```

---

## 性能优化

### 1. 并行标注（实验性）

```python
from concurrent.futures import ThreadPoolExecutor
import json

def annotate_sample(sample):
    annotator = VLMStyleAnnotator(...)
    return annotator.annotate_single_sample(**sample)

with open('data/tbu/tbu_for_annotation.json', 'r') as f:
    samples = json.load(f)

# 并行标注（注意 API 限流）
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(annotate_sample, samples[:100]))
```

### 2. 批量保存优化

默认每 10 个样本保存一次，可以根据需要调整：

```bash
# 更频繁保存（更安全但慢）
--save_interval 5

# 更少保存（更快但中断风险大）
--save_interval 50
```

---

## 故障排查

### 问题 1: API 连接失败

```
错误: Connection error
```

**解决方案**:
1. 检查网络连接
2. 确认 API 地址正确
3. 验证 API 密钥有效

```bash
# 测试 API 连接
curl -H "x-api-key: $ANTHROPIC_AUTH_TOKEN" \
     $ANTHROPIC_BASE_URL/v1/messages
```

### 问题 2: 解析响应失败

```
错误: 解析响应失败: ...
```

**原因**: VLM 输出格式不符合预期

**解决方案**:
1. 查看 `raw_response` 字段
2. 调整提示词
3. 检查模型版本

### 问题 3: 内存不足

```
错误: CUDA out of memory
```

**解决方案**:
```bash
# 减小 batch size
--batch_size 32

# 使用 CPU
--device cpu
```

### 问题 4: 标注中断

**解决方案**: 使用断点续传

```bash
# 查看已标注数量
python -c "
import json
with open('data/tbu/tbu_annotated.json', 'r') as f:
    data = json.load(f)
print(f'已标注: {len(data)} 个样本')
"

# 从中断处继续
python annotate_tbu.py annotate \
    --input data/tbu/tbu_for_annotation.json \
    --output data/tbu/tbu_annotated.json \
    --resume 1234  # 替换为实际数量
```

---

## 预期结果

### 标注质量

基于 VLM 的标注预期质量：

| 指标 | 预期值 |
|------|--------|
| 平均置信度 | 0.70 - 0.85 |
| 高置信度样本 (>0.8) | 40-60% |
| 低置信度样本 (<0.5) | 5-15% |
| 标签一致性 | 75-85% |

### 训练效果

使用 VLM 标注数据训练的模型：

| 数据集规模 | 训练时间 | 准确率 |
|-----------|---------|--------|
| 1,000 样本 | 10 分钟 | 50-60% |
| 5,000 样本 | 30 分钟 | 60-70% |
| 10,000 样本 | 1 小时 | 65-75% |
| 37,000 样本 | 2-3 小时 | 70-80% |

---

## 文件结构

```
teacher_style_analysis/
├── src/
│   └── annotation/
│       ├── vlm_annotator.py       # VLM 标注器核心逻辑
│       └── convert_tbu.py         # TBU 数据转换
├── annotate_tbu.py                # 批量标注脚本
├── annotate_and_train_tbu.sh      # 一键流程脚本
├── test_vlm_annotator.sh          # 测试脚本
├── VLM_ANNOTATION_GUIDE.md        # 本文档
└── data/
    └── tbu/
        ├── tbu_for_annotation.json    # 转换后的数据
        ├── tbu_annotated.json         # 标注结果
        └── tbu_training.json          # 训练数据
```

---

## 下一步

1. ✅ 测试 VLM 标注器
2. ✅ 小规模测试（10-100 样本）
3. ⏳ 全量标注（37,000 样本）
4. ⏳ 训练模型
5. ⏳ 评估和优化

需要帮助？查看相关文档：
- `RECOMMENDED_DATASETS.md` - 数据集推荐
- `MM-TBA_PROBLEM_ANALYSIS.md` - 数据质量分析
- `DEEP_LEARNING_INTEGRATION.md` - 深度学习集成
