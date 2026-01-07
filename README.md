# 教师教学风格分析系统

基于深度学习的多模态教学风格识别与分析系统

## 📋 项目概述

本系统使用多模态深度学习方法，从课堂教学视频中自动识别和分析教师的教学风格。系统支持7种教学风格分类：

1. **理论讲授型** - 系统讲解理论知识
2. **启发引导型** - 提问引导学生思考
3. **互动导向型** - 强调师生互动参与
4. **逻辑推导型** - 注重逻辑推理过程
5. **题目驱动型** - 以解题为核心教学
6. **情感表达型** - 善用肢体语言感染学生
7. **耐心细致型** - 讲解细致关注细节

## 🚀 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置CUDA（如使用GPU）
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.7
```

### 分析单个视频

```bash
python -m src.main analyze \
    --video path/to/video.mp4 \
    --teacher T001 \
    --discipline 数学 \
    --grade 初中 \
    --mode deep_learning \
    --device cuda
```

### 启动API服务

```bash
python -m src.main server --host 0.0.0.0 --port 8000
# 访问 http://localhost:8000/docs 查看API文档
```

## 🎯 主要功能

### 1. 多模态特征提取
- 视频特征：关键帧提取、时序分析
- 音频特征：语音转文本（Whisper）、声学特征
- 文本特征：BERT嵌入、NLP分析
- 姿态估计：MediaPipe姿态关键点
- 物体检测：YOLO课堂物体识别

### 2. 深度学习模型 (MMAN)
- **架构**：Multi-Modal Attention Network
- **组件**：
  - 模态编码器：统一多模态表示
  - Transformer：跨模态注意力机制
  - BiLSTM：时序特征建模
  - 注意力池化：加权特征聚合

### 3. VLM大规模自动标注
使用Claude Vision API进行大规模数据集自动标注，支持：
- MM-TBA数据集标注
- 断点续传
- 批量处理

## 📊 数据集与模型训练

### MM-TBA数据集训练

所有MM-TBA数据处理工具位于：`data/mm-tba/tools/`

**快速开始**：
```bash
cd data/mm-tba/tools
bash train_mmtba_gpu.sh
```

**方法1：使用VLM自动标注（推荐）**

```bash
cd data/mm-tba/tools

# 1. 转换数据为标注格式
python convert_mmtba_vlm.py to-annotation \
    --mmtba_path ../MM-TBA \
    --output ../for_vlm_annotation.json

# 2. VLM标注（需配置API）
export ANTHROPIC_BASE_URL="https://aidev.deyecloud.com/api"
export ANTHROPIC_AUTH_TOKEN="your_token"

python -m src.annotation.vlm_annotator \
    --input ../for_vlm_annotation.json \
    --output ../vlm_annotated.json \
    --save_interval 10

# 3. 转换为训练格式
python convert_mmtba_vlm.py to-training \
    --input ../vlm_annotated.json \
    --output ../training.json

# 4. 训练模型
bash train_mmtba_gpu.sh
```

**方法2：使用原始标注**

```bash
cd data/mm-tba/tools

# 1. 直接转换数据集
python convert_mmtba.py \
    --mmtba_path ../MM-TBA \
    --output ../mmtba_converted.json

# 2. 训练模型
python -m src.models.deep_learning.train \
    --data_path ../mmtba_converted.json \
    --batch_size 64 \
    --num_epochs 150 \
    --device cuda \
    --checkpoint_dir ../../../checkpoints/mm-tba
```

详细说明请查看：`data/mm-tba/tools/README.md`

### 模型配置选项

- `default`: 平衡性能和精度
- `lightweight`: 快速推理，低资源
- `high_accuracy`: 最高精度，高计算

## 📂 项目结构

```
teacher_style_analysis/
├── README.md               # 项目主文档
├── docs/                   # 详细文档
│   ├── CLAUDE.md          # Claude Code开发指南
│   ├── README_VLM.md      # VLM标注说明
│   └── MM-TBA_TRAINING_GUIDE.md
├── src/
│   ├── main.py            # 主入口
│   ├── features/          # 特征提取模块
│   │   ├── feature_extractor.py
│   │   ├── video_feature_extractor.py
│   │   ├── audio_feature_extractor.py
│   │   ├── text_feature_extractor.py
│   │   ├── pose_estimation.py
│   │   └── multimodal_fusion.py
│   ├── models/
│   │   ├── deep_learning/ # MMAN模型
│   │   │   ├── mman_model.py
│   │   │   ├── train.py
│   │   │   ├── trainer.py
│   │   │   └── dataset.py
│   │   ├── core/
│   │   │   └── style_classifier.py
│   │   └── weights/       # 预训练模型权重
│   ├── annotation/        # VLM标注系统
│   │   └── vlm_annotator.py
│   └── api/               # API服务
├── data/
│   └── mm-tba/            # MM-TBA数据集
│       ├── MM-TBA/        # 原始数据
│       ├── tools/         # ⭐ 数据处理工具
│       │   ├── README.md
│       │   ├── convert_mmtba.py
│       │   ├── convert_mmtba_vlm.py
│       │   ├── train_mmtba_gpu.sh
│       │   └── train_gpu.sh
│       ├── for_vlm_annotation.json
│       ├── vlm_annotated.json
│       └── training.json
├── checkpoints/           # 模型检查点
└── logs/                  # 训练日志
```

## 🔧 分类模式

### 1. 深度学习模式（推荐）
```bash
--mode deep_learning
```
使用训练好的MMAN模型进行分类，准确率最高。

### 2. 规则模式
```bash
--mode rule
```
基于手工特征和规则阈值，适合快速分析。

### 3. 混合模式
```bash
--mode hybrid
```
结合深度学习和规则系统的优势。

## 📝 常用命令

### 批量分析
```bash
python -m src.main batch \
    --dir path/to/videos/ \
    --teacher T001 \
    --discipline 数学 \
    --grade 初中 \
    --device cuda
```

### 导出结果
```bash
python -m src.main export \
    --video_id xxx \
    --format json  # json/csv/excel
```

### 系统状态检查
```bash
python -m src.main status
```

## 🔬 技术栈

- **深度学习**: PyTorch, Transformers
- **特征提取**:
  - OpenCV (视频处理)
  - Whisper (语音识别)
  - BERT (文本理解)
  - MediaPipe (姿态估计)
  - YOLOv8 (物体检测)
- **API服务**: FastAPI, Uvicorn
- **VLM标注**: Claude API (Anthropic)

## 📖 详细文档

- `docs/CLAUDE.md` - Claude Code开发指南
- `docs/README_VLM.md` - VLM标注系统详细说明
- `docs/MM-TBA_TRAINING_GUIDE.md` - MM-TBA数据集训练指南
- `docs/DEEP_LEARNING_INTEGRATION.md` - 深度学习模块集成文档
- `data/mm-tba/tools/README.md` - MM-TBA数据处理工具说明

## ⚠️ 重要说明

1. **MediaPipe版本**: 本项目使用MediaPipe 0.10+新API，不兼容旧版本
2. **GPU支持**: 推荐使用CUDA 11.7，其他版本需修改环境变量
3. **API配置**: VLM标注需要配置内部API端点和认证令牌
4. **模型权重**: 首次运行会自动下载必要的预训练模型

## 🤝 贡献

本项目为研究项目，用于教学风格分析的学术研究。

## 📄 许可

本项目仅用于学术研究目的。

---

**更新日期**: 2026-01-07
**版本**: v1.0
