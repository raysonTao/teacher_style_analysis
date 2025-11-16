# 教师教学风格分析系统

基于多模态特征融合的智能教学风格识别与个性化反馈系统

## 项目简介

本项目是一个基于人工智能技术的教师教学风格分析系统，通过分析教学视频中的多模态数据（视频、音频、文本），识别教师的教学风格类型，并提供个性化的改进建议。系统采用了论文中提出的 CMAT（Combined Multi-modal Analysis and Teaching）模型，实现了规则驱动与机器学习相结合的风格分类方法，并创新性地提出了风格匹配度指数（SMI）来评估教学风格与学科需求的匹配程度。

## 系统架构

系统由以下核心模块组成：

1. **特征提取模块**：从视频、音频和文本中提取多模态特征
2. **风格识别模块**：使用 CMAT 模型进行教师风格分类和可解释性分析
3. **个性化反馈模块**：计算 SMI 和生成改进建议
4. **API 服务模块**：提供系统接口层
5. **实验模块**：实现论文中描述的实验方法

## 目录结构

```
teacher_style_analysis/
├── features/            # 特征提取模块
│   ├── __init__.py
│   └── feature_extractor.py  # 多模态特征提取器
├── models/              # 模型模块
│   ├── __init__.py
│   └── style_classifier.py   # 风格分类器
├── feedback/            # 反馈模块
│   ├── __init__.py
│   └── feedback_generator.py # 反馈生成器
├── api/                 # API 模块
│   ├── __init__.py
│   └── api_handler.py        # API 处理器
├── experiments/         # 实验模块
│   ├── __init__.py
│   ├── configs/         # 实验配置
│   ├── data/            # 实验数据
│   ├── models/          # 实验模型
│   ├── results/         # 实验结果
│   └── visualizations/  # 结果可视化
├── tests/               # 测试模块
│   ├── __init__.py
│   ├── test_feature_extractor.py
│   ├── test_style_classifier.py
│   ├── test_feedback_generator.py
│   ├── test_api_handler.py
│   ├── test_main.py
│   ├── test_integration.py
│   └── run_tests.py
├── main.py              # 主入口
├── requirements.txt     # 依赖
└── README.md            # 说明文档
```

## 安装与配置

### 1. 环境要求

- Python 3.8+
- CUDA 11.7+ (推荐，用于 GPU 加速)

### 2. 安装依赖

```bash
cd /Users/rayson/Documents/毕业大论文/05_code/teacher_style_analysis
pip install -r requirements.txt
```

### 3. 预训练模型下载

系统使用了以下预训练模型，会在首次运行时自动下载：

- YOLOv8：用于人体检测和姿态估计
- Whisper：用于语音识别
- BERT：用于文本特征提取

## 使用方法

### 1. 命令行运行

可以通过 `main.py` 运行系统的主要功能：

```bash
# 单个视频分析
python main.py --video_path path/to/video.mp4 --teacher_id teacher1 --subject "数学" --output_dir results/

# 批量分析
python main.py --batch_mode --directory path/to/videos/ --teacher_id teacher1 --subject "数学" --output_dir results/

# 运行实验
python main.py --run_experiment model_comparison --output_dir experiments/results/
```

### 2. 启动 API 服务

```bash
python -m api.api_handler
```

API 服务将在 http://localhost:8000 启动，提供以下端点：

- `POST /api/upload_video`：上传教学视频
- `POST /api/analyze_style/{video_id}`：分析教学风格
- `GET /api/videos`：获取视频列表
- `GET /api/videos/{video_id}`：获取视频详情
- `GET /api/teachers/{teacher_id}/growth`：获取教师成长轨迹
- `DELETE /api/videos/{video_id}`：删除视频
- `GET /api/health`：健康检查
- `GET /api/config`：获取系统配置

### 3. 运行测试

```bash
# 运行所有测试
python -m tests.run_tests

# 运行特定测试文件
python -m tests.run_tests -f test_feature_extractor.py

# 生成测试报告
python -m tests.run_tests -r test_report.md
```

## 实验模块

实验模块实现了论文中描述的所有实验方法：

1. **模型性能比较实验**：比较不同分类算法在教师风格识别任务上的性能
2. **多模态融合效果实验**：评估不同模态特征组合对分类性能的影响
3. **规则与机器学习融合实验**：研究 lambda 权重对融合效果的影响
4. **SMI 验证实验**：验证风格匹配度指数（SMI）计算方法的有效性
5. **跨学科适应性实验**：评估模型在不同学科教学视频上的泛化能力

可以运行以下命令执行实验：

```bash
# 运行所有实验
python -m experiments.models.model_comparator
python -m experiments.models.multimodal_fusion_experiment
python -m experiments.models.rule_ml_fusion_experiment
python -m experiments.models.smi_validation_experiment
python -m experiments.models.cross_discipline_experiment

# 分析实验结果
python -m experiments.visualizations.result_analyzer
```

## 系统功能特点

1. **多模态特征融合**：整合视频、音频和文本特征，提供更全面的分析
2. **规则驱动与机器学习结合**：采用 CMAT 模型提高分类准确性
3. **风格匹配度指数（SMI）**：创新性地评估教学风格与学科的匹配程度
4. **个性化反馈**：基于分析结果生成针对性的改进建议
5. **教学成长跟踪**：记录历史数据，分析教师教学风格的变化趋势
6. **可解释性分析**：提供分类结果的可视化和解释

## 注意事项

1. 系统首次运行会下载预训练模型，可能需要一定时间
2. 视频分析需要较大的计算资源，推荐使用 GPU 加速
3. 对于大规模视频数据，建议使用批量分析功能
4. 实验数据默认使用合成数据，可根据需要修改为真实数据集

## 许可证

本项目仅供学术研究使用。

## 联系信息

如有问题或建议，请联系项目开发者。