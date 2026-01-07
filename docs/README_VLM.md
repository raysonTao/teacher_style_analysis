# TBU 数据集 VLM 标注系统

使用公司内部 Claude API 对 TBU 数据集进行教学风格自动标注。

## 快速开始

### 前置要求

```bash
# 安装依赖
pip install anthropic tqdm
```

### 一键标注和训练

```bash
# 小规模测试（10个样本，约3分钟）
./annotate_and_train_tbu.sh data/TBU 10

# 中等规模（1000个样本，约5小时）
./annotate_and_train_tbu.sh data/TBU 1000

# 全量标注（37000个样本，约175小时）
./annotate_and_train_tbu.sh data/TBU
```

脚本会自动完成：
1. 转换 TBU 数据为标注格式
2. VLM 批量标注教学风格
3. 转换为训练格式
4. 训练深度学习模型

## 配置

系统已配置公司内部 API：
- API 地址: `https://aidev.deyecloud.com/api`
- 模型: `claude-opus-4-5-20251101`
- 环境变量已在脚本中设置

## 分步执行

### 1. 转换 TBU 数据

```bash
python -m src.annotation.convert_tbu \
    --tbu_path data/TBU \
    --output data/tbu/tbu_for_annotation.json
```

### 2. 批量标注

```bash
python annotate_tbu.py annotate \
    --input data/tbu/tbu_for_annotation.json \
    --output data/tbu/tbu_annotated.json \
    --save_interval 10
```

常用参数：
- `--max_samples 1000` - 限制标注数量
- `--resume 500` - 从第500个样本继续（断点续传）
- `--save_interval 10` - 每10个样本保存一次

### 3. 转换为训练格式

```bash
python annotate_tbu.py convert \
    --input data/tbu/tbu_annotated.json \
    --output data/tbu/tbu_training.json
```

### 4. 训练模型

```bash
python -m src.models.deep_learning.train \
    --data_path data/tbu/tbu_training.json \
    --batch_size 64 \
    --num_epochs 150 \
    --device cuda \
    --checkpoint_dir ./checkpoints/tbu
```

## 成本估算

```bash
python estimate_cost.py
```

基于 Opus 4.5 定价（$5/$25 per Mtok）：

| 样本数 | 成本(USD) | 成本(CNY) | 时间 |
|--------|----------|-----------|------|
| 100 | $1 | ¥7 | 0.5小时 |
| 1,000 | $10 | ¥72 | 5小时 |
| 5,000 | $50 | ¥360 | 24小时 |
| 37,000 | $370 | ¥2,664 | 175小时 |

## 断点续传

如果标注中断：

```bash
# 查看已标注数量
python -c "import json; print(len(json.load(open('data/tbu/tbu_annotated.json'))))"

# 从中断处继续
python annotate_tbu.py annotate \
    --input data/tbu/tbu_for_annotation.json \
    --output data/tbu/tbu_annotated.json \
    --resume 1234  # 替换为实际数量
```

## 质量监控

```python
import json
from collections import Counter

with open('data/tbu/tbu_annotated.json', 'r') as f:
    data = json.load(f)

# 置信度统计
confidences = [s['annotation']['confidence'] for s in data]
print(f"平均置信度: {sum(confidences)/len(confidences):.3f}")
print(f"高置信度 (>0.8): {sum(1 for c in confidences if c > 0.8)}")

# 风格分布
styles = [s['annotation']['style'] for s in data]
print(Counter(styles))
```

## 使用训练好的模型

```bash
python -m src.main analyze \
    --video your_video.mp4 \
    --teacher_id T001 \
    --discipline 数学 \
    --grade 初中 \
    --mode deep_learning \
    --dl_checkpoint ./checkpoints/tbu/best_model.pth \
    --device cuda
```

## 7种教学风格

1. **理论讲授型** - 系统讲解理论知识
2. **启发引导型** - 提问引导学生思考
3. **互动导向型** - 强调师生互动参与
4. **逻辑推导型** - 注重逻辑推理过程
5. **题目驱动型** - 以解题为核心教学
6. **情感表达型** - 善用肢体语言感染学生
7. **耐心细致型** - 讲解细致关注细节

## 目录结构

```
teacher_style_analysis/
├── src/annotation/
│   ├── vlm_annotator.py      # VLM 标注器核心
│   └── convert_tbu.py         # TBU 数据转换
├── annotate_tbu.py            # 批量标注脚本
├── annotate_and_train_tbu.sh  # 一键流程脚本
├── estimate_cost.py           # 成本估算器
└── README_VLM.md              # 本文档
```

## 常见问题

**Q: 如何降低成本？**
A: 可以考虑使用更便宜的模型（如果公司支持）：
- Sonnet 4.5: 节省 40%
- Haiku 4.5: 节省 80%

**Q: 标注质量如何？**
A: 测试显示平均置信度约 85-90%，建议每1000个样本人工抽查10个。

**Q: 标注速度太慢？**
A: 平均 17秒/样本是正常速度。可以分批运行，或夜间标注。

---

**模型**: claude-opus-4-5-20251101
**状态**: 生产就绪
**更新**: 2026-01-05
