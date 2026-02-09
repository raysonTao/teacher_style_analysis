# 论文架构图生成指南

本目录包含论文所需的三个核心架构图的 LaTeX/TikZ 源代码。

## 📊 生成的架构图

### 1. **SHAPE网络架构图** (`shape_architecture.tex`)
- **对应论文位置**: 图3.2（Section 3.3.2）
- **内容**: SHAPE (Semantic Hierarchical Attention Profiling Engine) 详细网络结构
- **核心展示**:
  - 输入特征 (20+15+35维)
  - 特征投影层 (512×3维)
  - **跨模态注意��层** (核心创新)
  - BiLSTM时序建模 (1024维)
  - 注意力池化
  - 7类风格分类器

### 2. **系统四层架构图** (`system_architecture.tex`)
- **对应论文位置**: 图3.1（Section 3.1.2）
- **内容**: 完整系统的四层架构设计
- **核心展示**:
  - Layer 1: 数据管理层 (MySQL + Redis + MinIO)
  - Layer 2: 特征提取层 (视频/音频/文本Pipeline并行)
  - Layer 3: 模型推理层 (SHAPE + SHAP)
  - Layer 4: 应用服务层 (画像生成 + 可视化)

### 3. **多模态特征提取流程图** (`multimodal_pipeline.tex`)
- **对应论文位置**: 图5-1（Section 5.2.1，可选）
- **内容**: Algorithm 1的可视化流程
- **核心展示**:
  - 视频流水线: YOLOv8 → DeepSORT → MediaPipe → ST-GCN (0.82s)
  - 音频流水线: Whisper → Wav2Vec2 → 情感分类 (0.37s)
  - 文本流水线: BERT → H-DAR → NLP统计 (0.15s)

---

## 🛠️ 编译方法

### 方法1: 本地编译（需要安装 LaTeX）

#### 安装 LaTeX（如果未安装）
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS (使用 Homebrew)
brew install --cask mactex

# 或者安装轻量版
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-pictures
```

#### 编译单个图
```bash
cd doc/08版本/架构图生成

# 编译SHAPE网络架构图
pdflatex shape_architecture.tex

# 编译系统四层架构图
pdflatex system_architecture.tex

# 编译多模态流程图
pdflatex multimodal_pipeline.tex
```

#### 编译所有图
```bash
make all
# 或
./compile_all.sh
```

### 方法2: 在线编译（无需安装）

推荐使用在线 LaTeX 编译器：

1. **Overleaf** (推荐)
   - 访问 https://www.overleaf.com/
   - 创建新项目 → 上传 `.tex` 文件
   - 点击 "Recompile" ���动生成 PDF

2. **LaTeX.Online**
   - 访问 https://latexonline.cc/
   - 上传 `.tex` 文件
   - 自动编译并下载 PDF

### 方法3: 使用 Docker

如果不想安装完整的 LaTeX 环境：

```bash
# 使用官方 LaTeX Docker 镜像
docker run --rm -v $(pwd):/workdir texlive/texlive pdflatex shape_architecture.tex
```

---

## 📤 生成 PNG/高分辨率图片

编译成功后，可以转换为高分辨率图片：

### 使用 ImageMagick（推荐）
```bash
# 安装 ImageMagick
sudo apt-get install imagemagick

# 转换为300 DPI的PNG
convert -density 300 shape_architecture.pdf -quality 90 shape_architecture.png
convert -density 300 system_architecture.pdf -quality 90 system_architecture.png
convert -density 300 multimodal_pipeline.pdf -quality 90 multimodal_pipeline.png

# 或更高分辨率（600 DPI，用于印刷）
convert -density 600 shape_architecture.pdf -quality 100 shape_architecture_hd.png
```

### 使用 pdftoppm
```bash
# 安装 poppler-utils
sudo apt-get install poppler-utils

# 转换为PNG
pdftoppm -png -r 300 shape_architecture.pdf shape_architecture
```

---

## 📝 插入论文

编译生成PDF后，在论文中插入：

### LaTeX论文中
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{doc/08版本/架构图生成/shape_architecture.pdf}
    \caption{SHAPE网络架构图 (Semantic Hierarchical Attention Profiling Engine)}
    \label{fig:shape_architecture}
\end{figure}
```

### Word论文中
1. 生成PNG图片（使用上述方法）
2. 插入 → 图片 → 选择PNG文件
3. 调整大小，添加图注

---

## 🎨 自定义修改

### 修改颜色
在 `.tex` 文件中修改颜色定义：
```latex
\definecolor{video}{RGB}{255,127,127}  % 视觉模态 - 红色系
\definecolor{audio}{RGB}{127,255,127}  % 音频模态 - 绿色系
\definecolor{text}{RGB}{127,127,255}   % 文本模态 - 蓝色系
```

### 修改布局
调整 `node distance` 参数：
```latex
node distance=1.5cm and 2cm,  % 垂直间距1.5cm, 水平间距2cm
```

### 修改字体大小
```latex
font=\small    % 改为 \footnotesize (更小) 或 \large (更大)
```

---

## 🐛 常见问题

### 1. 编译错误：`! LaTeX Error: File 'tikz.sty' not found`
**解决方法**: 安装缺失的包
```bash
sudo apt-get install texlive-pictures texlive-latex-extra
```

### 2. 中文显示乱码
**解决方法**: 使用 XeLaTeX 编译
```bash
xelatex shape_architecture.tex
```

### 3. 图片太大或太小
**解决方法**: 修改 `minimum width` 和 `minimum height` 参数
```latex
layer/.style={..., minimum width=14cm, minimum height=2.5cm}
```

### 4. ImageMagick 转换失败
**解决方法**: 修改 ImageMagick 安全策略
```bash
sudo nano /etc/ImageMagick-6/policy.xml
# 找到这行：<policy domain="coder" rights="none" pattern="PDF" />
# 改为：<policy domain="coder" rights="read|write" pattern="PDF" />
```

---

## 📦 文件清单

```
架构图生成/
├── shape_architecture.py          # 生成脚本
├── shape_architecture.tex         # SHAPE网络架构图源码
├── system_architecture.tex        # 系统四层架构图源码
├── multimodal_pipeline.tex        # 多模态流程图源码
├── README.md                      # 本说明文档
├── compile_all.sh                 # 批量编译脚本（待生成）
└── Makefile                       # Make编译配置（待生成）
```

---

## 🎯 快速开始

**最快方法** - 使用 Overleaf 在线编译：

1. 访问 https://www.overleaf.com/
2. 点击 "New Project" → "Upload Project"
3. 上传 `shape_architecture.tex`
4. 等待自动编译完成
5. 下载 PDF

**本地编译** - 仅需3条命令：

```bash
cd doc/08版本/架构图生成
pdflatex shape_architecture.tex
convert -density 300 shape_architecture.pdf shape_architecture.png
```

---

## 📧 技术支持

如遇到问题：
1. 检查 LaTeX 日志文件 (`*.log`)
2. 确认安装了所需的包 (tikz, xcolor, amsmath)
3. 尝试使用在线编译器（Overleaf）

---

**生成时间**: 2026-02-07
**对应论文**: 基于课堂录像的教师风格画像分析系统
**版本**: 08稿
