# 教师教学风格画像分析系统

基于多模态特征融合的智能教学风格识别与个性化反馈系统

## 项目简介

本项目是一个基于人工智能技术的教师教学风格画像分析系统，通过分析教学视频中的多模态数据（音频、文本、教学行为），自动识别教师的教学风格类型，并生成个性化的教学改进建议。

系统核心创新点包括：
- 实现论文提出的 CMAT（Combined Multi-modal Attention-based Teaching style model）模型，结合规则驱动与机器学习的双层次分类架构
- 提出风格匹配度指数（SMI）来评估教学风格与学科/年级理想标准的匹配程度
- 支持9个学科（数学、语文等）和3个学段（初中、高中、大学）的差异化评估标准
- 提供教学成长轨迹分析，追踪教师专业发展趋势
- 生成具有可解释性的个性化反馈报告

## 系统架构

系统采用模块化分层架构，包含以下核心组件：

1. **特征提取模块**：从教学视频中提取多模态特征（音频特征、文本特征、教学行为特征）
2. **CMAT 模型模块**：实现双层次分类架构（规则驱动层 + 机器学习层），识别7种教学风格
3. **个性化反馈模块**：计算风格匹配度指数(SMI)，生成针对性改进建议和教学成长分析
4. **API 服务模块**：提供RESTful接口，支持视频上传、风格分析、结果查询等功能
5. **数据管理模块**：处理文件存储、元数据管理和结果缓存
6. **实验模块**：实现论文中的对比实验和性能验证

## 目录结构

```
src/
├── features/            # 特征提取模块
│   └── feature_extractor.py  # 多模态特征提取器，处理音频、文本和教学行为特征
├── models/              # 模型模块
│   ├── cmat_model.pkl   # 预训练的CMAT模型
│   └── style_classifier.py   # 风格分类器，实现CMAT双层次架构
├── feedback/            # 反馈模块
│   └── feedback_generator.py # 反馈生成器，计算SMI和生成改进建议
├── api/                 # API 模块
│   └── api_handler.py        # API 处理器，提供RESTful接口
├── data/                # 数据存储目录
│   ├── audio/           # 提取的音频文件
│   ├── text/            # 提取的文本内容
│   ├── videos/          # 上传的教学视频
│   ├── results/         # 分析结果
│   ├── extracted_features/  # 提取的特征
│   └── data_manager.py  # 数据管理模块
├── experiments/         # 实验模块（论文实验实现）
│   ├── configs/         # 实验配置文件
│   ├── data/            # 实验数据集
│   ├── models/          # 实验模型实现
│   ├── results/         # 实验结果存储
│   └── visualizations/  # 结果可视化脚本
├── tests/               # 测试模块
│   ├── test_feature_extractor.py  # 特征提取器测试
│   ├── test_style_classifier.py   # 风格分类器测试
│   ├── test_feedback_generator.py # 反馈生成器测试
│   ├── test_api_handler.py        # API处理器测试
│   ├── test_main.py              # 主程序测试
│   ├── test_integration.py       # 集成测试
│   └── run_tests.py              # 测试运行脚本
├── config/              # 配置目录
│   └── config.py        # 系统配置文件
├── exports/             # 导出结果目录
├── main.py              # 系统主入口
├── requirements.txt     # Python依赖列表
└── README.md            # 项目说明文档
```

## 安装与配置

### 1. 环境要求

- Python 3.8+ 或更高版本
- 推荐使用虚拟环境（如 conda 或 venv）
- CUDA 11.7+（推荐，用于 GPU 加速，可大幅提升视频分析速度）
- 至少 8GB RAM，推荐 16GB+
- 足够的磁盘空间存储视频和分析结果

### 2. 安装步骤

#### 2.1 创建虚拟环境（推荐）

```bash
# 使用 conda 创建虚拟环境
conda create -n teacher_style python=3.9
conda activate teacher_style

# 或者使用 venv
python -m venv teacher_style_env
source teacher_style_env/bin/activate  # Linux/Mac
tteacher_style_env\Scripts\activate  # Windows
```

#### 2.2 安装依赖包

```bash
cd /path/to/your/project
pip install -r requirements.txt
```

#### 2.3 安装FFmpeg

FFmpeg是本项目音频处理的必要依赖，用于从视频中提取音频和音频格式转换。请根据您的操作系统选择安装方式：

##### macOS
```bash
# 使用 Homebrew 安装（推荐）
brew install ffmpeg

# 或者使用 MacPorts
port install ffmpeg
```

##### Linux（Ubuntu/Debian）
```bash
# 更新软件源
sudo apt-get update

# 安装FFmpeg
sudo apt-get install ffmpeg
```

##### Linux（CentOS/RHEL）
```bash
# 安装EPEL仓库
sudo yum install epel-release

# 安装Nux Dextop仓库
sudo rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm

# 安装FFmpeg
sudo yum install ffmpeg ffmpeg-devel
```

##### Windows

1. 访问FFmpeg官方网站：https://ffmpeg.org/download.html
2. 下载Windows版本的FFmpeg
3. 解压文件并将bin目录添加到系统环境变量PATH中
4. 重启命令提示符或终端

##### 验证安装

安装完成后，您可以通过以下命令验证FFmpeg是否正确安装：

```bash
ffmpeg -version
```

如果安装成功，您将看到FFmpeg的版本信息。

#### 2.4 预训练模型下载

系统使用了以下预训练模型，会在首次运行时自动下载：

- Whisper：用于高质量语音识别和情感分析
- BERT：用于文本特征提取和语义分析
- 自定义的 CMAT 模型：用于教学风格分类

### 3. 配置文件说明

主要配置文件位于 `config/config.py`，可根据需要修改以下关键参数：

- API 服务端口和主机地址
- 文件存储路径
- 分析参数配置（如音频采样率、文本处理参数等）
- 模型参数设置（如规则权重、阈值等）

### 4. 数据目录初始化

首次运行时，系统会自动创建所需的数据目录结构。如需手动创建：

```bash
mkdir -p data/videos data/audio data/text data/results data/extracted_features\mkdir -p exports
```

## 使用方法

### 1. 命令行运行

通过 `main.py` 可以执行系统的主要功能：

#### 1.1 单个视频分析

```bash
# 基本用法
python main.py analyze --video data/videos/2027张宇考研数学\ 基础30讲\ 导学.mp4 --teacher zhangyu --discipline "数学" --grade "大学"

# 完整参数示例
python main.py analyze \  
  --video "data/sample_lesson.mp4" \  
  --teacher "teacher001" \  
  --discipline "数学" \  
  --grade "高中"
```

参数说明：
- `--video`：教学视频文件路径（必需）
- `--teacher`：教师ID（必需）
- `--discipline`：学科（必需，支持：数学、语文、英语、物理、化学、生物、历史、地理、政治）
- `--grade`：年级（必需，支持：初中、高中、大学）

#### 1.2 批量视频分析

```bash
# 批量处理目录中的所有视频
python main.py batch \  
  --dir "data/videos/" \  
  --teacher "teacher001" \  
  --discipline "数学" \  
  --grade "高中"
```

参数说明：
- `--dir`：包含视频文件的目录路径（必需）
- `--teacher`：教师ID（必需）
- `--discipline`：学科（必需）
- `--grade`：年级（必需）

### 2. API 服务使用

#### 2.1 启动 API 服务

```bash
python -m api.api_handler
```

服务默认在 http://localhost:8000 启动。

#### 2.2 API 端点使用示例

**1. 上传教学视频**
```bash
curl -X POST "http://localhost:8000/api/upload_video" \
  -F "video=@/path/to/lesson.mp4" \
  -F "teacher_id=teacher001" \
  -F "discipline=数学" \
  -F "grade=高中"
```

**2. 分析教学风格**
```bash
curl -X POST "http://localhost:8000/api/analyze_style/video123"
```

**3. 获取视频列表**
```bash
curl -X GET "http://localhost:8000/api/videos?teacher_id=teacher001&page=1&page_size=10"
```

**4. 获取视频详情**
```bash
curl -X GET "http://localhost:8000/api/videos/video123"
```

**5. 获取教师成长轨迹**
```bash
curl -X GET "http://localhost:8000/api/teachers/teacher001/growth"
```

**6. 删除视频**
```bash
curl -X DELETE "http://localhost:8000/api/videos/video123"
```

**7. 系统健康检查**
```bash
curl -X GET "http://localhost:8000/api/health"
```

**8. 获取系统配置**
```bash
curl -X GET "http://localhost:8000/api/config"
```

### 3. 运行测试

```bash
# 运行所有测试
python -m tests.run_tests

# 运行特定测试文件
python -m tests.run_tests -f test_feature_extractor.py

# 生成测试报告
python -m tests.run_tests -r test_report.md
```

## 论文实验指南

### 1. 实验模块概述

实验模块实现了论文中描述的所有对比实验和性能验证方法，位于 `experiments/` 目录下。

### 2. 实验类型与目的

| 实验名称 | 实现文件 | 实验目的 | 论文章节 |
|---------|---------|---------|--------|
| 模型性能比较实验 | `experiments/models/model_comparator.py` | 比较CMAT模型与传统分类算法的性能差异 | 5.2 |
| 多模态融合效果实验 | `experiments/models/multimodal_fusion_experiment.py` | 验证不同模态特征组合的效果 | 5.3 |
| 规则与机器学习融合实验 | `experiments/models/rule_ml_fusion_experiment.py` | 研究lambda权重对分类性能的影响 | 5.4 |
| SMI验证实验 | `experiments/models/smi_validation_experiment.py` | 验证风格匹配度指数计算方法的有效性 | 5.5 |
| 跨学科适应性实验 | `experiments/models/cross_discipline_experiment.py` | 评估模型在不同学科上的泛化能力 | 5.6 |

### 3. 实验执行步骤

#### 3.1 准备实验数据

```bash
# 准备实验数据集（已包含在系统中）
# 数据集位于：experiments/data/
```

#### 3.2 执行单项实验

```bash
# 执行模型性能比较实验
python -m experiments.models.model_comparator

# 执行多模态融合效果实验
python -m experiments.models.multimodal_fusion_experiment

# 执行规则与机器学习融合实验
python -m experiments.models.rule_ml_fusion_experiment

# 执行SMI验证实验
python -m experiments.models.smi_validation_experiment

# 执行跨学科适应性实验
python -m experiments.models.cross_discipline_experiment
```

#### 3.3 执行完整实验流程

```bash
# 运行所有论文实验并生成完整报告
python -m experiments.models.run_all_experiments
```

#### 3.4 实验结果可视化

```bash
# 生成论文图表
python -m experiments.visualizations.paper_visualizations

# 生成实验报告
python -m experiments.visualizations.generate_experiment_report
```

### 4. 实验配置文件

实验参数可在 `experiments/configs/` 目录下配置：

- `experiment_config.py`：通用实验参数
- `model_config.py`：模型相关参数
- `fusion_config.py`：融合相关参数
- `visualization_config.py`：可视化参数

### 5. 论文图表生成

系统会自动生成论文中所需的图表，保存至 `experiments/visualizations/charts/` 目录：

- 模型性能对比图
- 多模态融合效果热力图
- Lambda权重影响曲线
- SMI验证散点图
- 跨学科性能对比柱状图

### 6. 实验结果导出

实验结果将自动保存至以下位置：

- 实验数据：`experiments/results/data/`
- 评估指标：`experiments/results/metrics/`
- 可视化图表：`experiments/visualizations/charts/`
- 完整报告：`experiments/results/full_report.md`

## 系统功能特点

### 1. 核心技术特性

- **多模态特征融合**：整合音频特征（语速、语调变化、情绪分数）、文本特征（词汇丰富度、提问频率、逻辑指示词）和教学行为特征，提供全面分析
- **CMAT双层次架构**：结合规则驱动层（基于教育专业知识）和机器学习层（数据驱动学习），兼顾准确性和可解释性
- **风格匹配度指数（SMI）**：创新性地提出0-100分的量化指标，评估教学风格与学科/年级理想标准的匹配程度
- **可解释性分析**：提供特征贡献度分析，解释分类结果和改进建议的决策依据

### 2. 教学评估功能

- **7种教学风格识别**：理论讲授型、启发引导型、互动交流型、实践操作型、情感感染型、结构严谨型、开放探索型
- **9大学科差异化标准**：数学、语文、英语、物理、化学、生物、历史、地理、政治
- **3个学段适配**：初中、高中、大学，考虑不同学段教学特点
- **个性化改进建议**：基于分析结果生成针对性的教学改进指导

### 3. 教师发展支持

- **教学成长轨迹分析**：追踪SMI指数变化和风格演变趋势
- **历史对比分析**：支持多次教学分析结果的纵向对比
- **专业发展报告**：生成包含优势、不足和发展建议的综合报告

## 系统使用注意事项

### 1. 系统要求

- 首次运行会自动下载预训练模型，建议在网络良好环境下进行
- 视频分析计算密集，推荐使用GPU加速（可提升5-10倍速度）
- 单个10分钟视频分析约需2-5分钟（取决于硬件配置）
- 批量处理时建议设置合理的并行参数，避免资源耗尽

### 2. 视频要求

- 支持MP4、AVI、MOV等常见视频格式
- 建议视频分辨率720p以上，保证音频质量清晰
- 教学视频应包含完整的教学过程，建议时长5-45分钟
- 多人教学场景建议确保主讲教师声音清晰可辨

### 3. 结果解读

- SMI分数（0-100）：越高表示与学科/年级理想标准匹配度越高
- 风格分类：系统会识别主导风格和次要风格，反映教学多样性
- 改进建议：基于最低分维度和特征贡献度生成，建议循序渐进实施
- 成长分析：需积累多次分析数据才能生成有意义的趋势图

## 许可证

本项目仅用于学术研究和论文撰写，未经授权不得用于商业用途。

## 联系信息

如有问题、建议或技术支持需求，请联系项目开发团队。

## 快速上手指南（论文撰写版）

### 步骤1：环境准备
```bash
# 创建虚拟环境
conda create -n teacher_style python=3.9
conda activate teacher_style

# 安装依赖
cd /path/to/your/project
pip install -r requirements.txt
```

### 步骤2：运行论文实验
```bash
# 执行完整实验流程
python -m experiments.models.run_all_experiments

# 生成论文图表
python -m experiments.visualizations.paper_visualizations
```

### 步骤3：准备演示案例
```bash
# 使用示例视频生成分析报告
python main.py analyze \
  --video "data/sample_lesson.mp4" \
  --teacher "demo_teacher" \
  --discipline "数学" \
  --grade "高中"
```

### 步骤4：启动API服务（可选）
```bash
# 启动Web服务进行交互式演示
python -m api.api_handler
```

## 论文完成流程

### 1. 数据收集与预处理
- 准备教学视频数据集
- 按照系统要求格式化元数据信息
- 确保视频质量符合分析标准

### 2. 实验执行
- 按照上述实验指南执行所有论文实验
- 生成实验结果和可视化图表
- 保存实验数据用于论文撰写

### 3. 系统演示
- 选择典型教学案例进行详细分析
- 生成教师风格画像报告
- 准备个性化反馈和教学改进建议示例

### 4. 论文撰写支持
- 从`experiments/results/full_report.md`获取实验分析结果
- 从`experiments/visualizations/charts/`获取论文图表
- 使用系统生成的教学风格分析报告作为案例展示

### 5. 论文图表来源说明
| 图表名称 | 生成脚本 | 文件位置 | 论文引用 |
|---------|---------|---------|--------|
| 模型性能对比图 | paper_visualizations.py | charts/model_comparison.png | 图5.1 |
| 多模态融合热力图 | paper_visualizations.py | charts/multimodal_fusion.png | 图5.2 |
| Lambda权重影响曲线 | paper_visualizations.py | charts/lambda_analysis.png | 图5.3 |
| SMI验证散点图 | paper_visualizations.py | charts/smi_validation.png | 图5.4 |
| 跨学科性能柱状图 | paper_visualizations.py | charts/cross_discipline.png | 图5.5 |

## 更新日志

### 1.0.0
- 初始版本，实现论文中提出的CMAT模型
- 支持9大学科和3个学段的教学风格分析
- 提供完整的API服务和实验模块