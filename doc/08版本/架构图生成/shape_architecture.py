#!/usr/bin/env python3
"""
SHAPE网络架构图生成脚本
使用 PlotNeuralNet 风格生成 TikZ 代码

论文：基于课堂录像的教师风格画像分析系统
Section 3.3: SHAPE (Semantic Hierarchical Attention Profiling Engine)
"""

import sys
sys.path.append('../')

# 导入 PlotNeuralNet 工具（如果可用）
# 如果没有安装，可以使用纯 TikZ 代码

def generate_shape_architecture_tikz():
    """
    生成 SHAPE 网络架构的 TikZ 代码

    架构流程：
    输入特征(20+15+35) → 特征投影(512×3) → 跨模态注意力(512×3)
    → BiLSTM(1024) → 注意力池化(1024) → 分类器(7)
    """

    tikz_code = r"""
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{amssymb}

\usetikzlibrary{positioning, arrows.meta, shapes.geometric, calc, decorations.pathreplacing}

% 定义颜色
\definecolor{video}{RGB}{255,127,127}      % 视觉模态 - 红色系
\definecolor{audio}{RGB}{127,255,127}      % 音频模态 - 绿色系
\definecolor{text}{RGB}{127,127,255}       % 文本模态 - 蓝色系
\definecolor{fusion}{RGB}{255,200,100}     % 融合层 - 橙色系
\definecolor{output}{RGB}{200,200,200}     % 输出层 - 灰色系

\begin{document}
\begin{tikzpicture}[
    node distance=1.5cm and 2cm,
    layer/.style={rectangle, draw, minimum width=2cm, minimum height=1.2cm, align=center, font=\small},
    modality/.style={rectangle, draw, minimum width=1.8cm, minimum height=1cm, align=center, font=\footnotesize, rounded corners=2pt},
    arrow/.style={-Stealth, thick},
    label/.style={font=\scriptsize, align=center}
]

% ==================== 第1层：输入特征 ====================
\node[modality, fill=video!30] (input_v) at (0,0) {视觉特征\\$F_v \in \mathbb{R}^{20}$};
\node[modality, fill=audio!30, below=0.8cm of input_v] (input_a) {音频特征\\$F_a \in \mathbb{R}^{15}$};
\node[modality, fill=text!30, below=0.8cm of input_a] (input_t) {文本特征\\$F_t \in \mathbb{R}^{35}$};

% 输入层标签
\node[label, above=0.3cm of input_v] {\textbf{输入层}\\(多模态特征)};

% ==================== 第2层：特征投影 ====================
\node[modality, fill=video!50, right=2.5cm of input_v] (proj_v) {投影层\\$F'_v \in \mathbb{R}^{512}$};
\node[modality, fill=audio!50, right=2.5cm of input_a] (proj_a) {投影层\\$F'_a \in \mathbb{R}^{512}$};
\node[modality, fill=text!50, right=2.5cm of input_t] (proj_t) {投影层\\$F'_t \in \mathbb{R}^{512}$};

% 投影层标签
\node[label, above=0.3cm of proj_v] {\textbf{模块1}\\特征投影};

% 投影箭头
\draw[arrow] (input_v) -- node[above, label] {$W_v, b_v$} (proj_v);
\draw[arrow] (input_a) -- node[above, label] {$W_a, b_a$} (proj_a);
\draw[arrow] (input_t) -- node[above, label] {$W_t, b_t$} (proj_t);

% ==================== 第3层：跨模态注意力 ====================
\node[modality, fill=fusion!30, right=2.5cm of proj_v] (attn_v) {注意力融合\\$\tilde{F}_v \in \mathbb{R}^{512}$};
\node[modality, fill=fusion!30, right=2.5cm of proj_a] (attn_a) {注意力融合\\$\tilde{F}_a \in \mathbb{R}^{512}$};
\node[modality, fill=fusion!30, right=2.5cm of proj_t] (attn_t) {注意力融合\\$\tilde{F}_t \in \mathbb{R}^{512}$};

% 注意力层标签
\node[label, above=0.3cm of attn_v] {\textbf{模块2}\\跨模态注意力};

% 跨模态注意力连接（核心创新）
% Video -> Audio, Text
\draw[arrow, bend left=15, color=video] (proj_v) to node[above, label, sloped] {\tiny $\alpha_{v \to a}$} (attn_a);
\draw[arrow, bend left=20, color=video] (proj_v) to node[above, label, sloped] {\tiny $\alpha_{v \to t}$} (attn_t);

% Audio -> Video, Text
\draw[arrow, bend left=15, color=audio] (proj_a) to node[above, label, sloped] {\tiny $\alpha_{a \to v}$} (attn_v);
\draw[arrow, bend left=15, color=audio] (proj_a) to node[below, label, sloped] {\tiny $\alpha_{a \to t}$} (attn_t);

% Text -> Video, Audio
\draw[arrow, bend left=20, color=text] (proj_t) to node[below, label, sloped] {\tiny $\alpha_{t \to v}$} (attn_v);
\draw[arrow, bend left=15, color=text] (proj_t) to node[below, label, sloped] {\tiny $\alpha_{t \to a}$} (attn_a);

% 残差连接
\draw[arrow, dashed, color=gray] (proj_v) -- (attn_v);
\draw[arrow, dashed, color=gray] (proj_a) -- (attn_a);
\draw[arrow, dashed, color=gray] (proj_t) -- (attn_t);

% ==================== 第4层：BiLSTM时序建模 ====================
\node[layer, fill=fusion!50, right=2.5cm of attn_a, minimum width=2.2cm, minimum height=2cm] (bilstm) {
    \textbf{BiLSTM}\\[0.2cm]
    $\overrightarrow{h}, \overleftarrow{h}$\\[0.1cm]
    $h \in \mathbb{R}^{1024}$
};

% BiLSTM标签
\node[label, above=0.3cm of bilstm] {\textbf{模块3}\\时序建模};

% 连接到BiLSTM
\draw[arrow] (attn_v) -| (bilstm.170);
\draw[arrow] (attn_a) -- (bilstm.180);
\draw[arrow] (attn_t) -| (bilstm.190);

% ==================== 第5层：注意力池化 ====================
\node[layer, fill=fusion!70, right=2cm of bilstm] (pool) {
    \textbf{注意力池化}\\[0.1cm]
    $h_{\text{pooled}}$\\[0.1cm]
    $\in \mathbb{R}^{1024}$
};

% 池化标签
\node[label, above=0.3cm of pool] {\textbf{模块4}\\特征聚合};

\draw[arrow] (bilstm) -- node[above, label] {$\beta_i$} (pool);

% ==================== 第6层：风格分类器 ====================
\node[layer, fill=output!70, right=2cm of pool, minimum width=2cm, minimum height=1.8cm] (classifier) {
    \textbf{分类器}\\[0.2cm]
    Softmax\\[0.1cm]
    $P(y) \in \mathbb{R}^7$
};

% 分类器标签
\node[label, above=0.3cm of classifier] {\textbf{模块5}\\风格预测};

\draw[arrow] (pool) -- (classifier);

% ==================== 输出 ====================
\node[label, right=0.5cm of classifier, align=left] (output) {
    \textbf{输出：}\\
    • 风格类别 $y$\\
    • 概率分布 $p$\\
    • 注意力权重 $\alpha$
};

\draw[arrow] (classifier) -- (output);

% ==================== 图例说明 ====================
\node[label, below=3cm of input_t, align=left, draw, dashed, minimum width=14cm, minimum height=1.5cm] (legend) {
    \textbf{图例说明：}\\
    \tikz \draw[arrow, color=video] (0,0) -- (0.3,0); 视觉模态连接 \quad
    \tikz \draw[arrow, color=audio] (0,0) -- (0.3,0); 音频模态连接 \quad
    \tikz \draw[arrow, color=text] (0,0) -- (0.3,0); 文本模态连接 \quad
    \tikz \draw[arrow, dashed, color=gray] (0,0) -- (0.3,0); 残差连接\\
    \textbf{核心创新：}跨模态注意力机制 ($\alpha_{i \to j}$) 实现模态间自适应交互，相比简单拼接准确率提升 \textbf{8.3\%}
};

% ==================== 整体标题 ====================
\node[above=0.8cm of input_v, font=\large\bfseries] {SHAPE网络架构图 (Semantic Hierarchical Attention Profiling Engine)};

\end{tikzpicture}
\end{document}
"""

    return tikz_code


def generate_system_architecture_tikz():
    """
    生成系统四层架构的 TikZ 代码

    四层架构：
    1. 数据管理层：MySQL + Redis + MinIO
    2. 特征提取层：视频/音频/文本并行处理
    3. 模型推理层：SHAPE + SHAP
    4. 应用服务层：画像生成 + 可视化
    """

    tikz_code = r"""
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{amsmath}

\usetikzlibrary{positioning, arrows.meta, shapes.geometric, calc, fit, backgrounds}

% 定义颜色
\definecolor{layer1}{RGB}{230,230,250}  % 数据层 - 淡紫色
\definecolor{layer2}{RGB}{255,228,181}  % 特征层 - 淡橙色
\definecolor{layer3}{RGB}{152,251,152}  % 推理层 - 淡绿色
\definecolor{layer4}{RGB}{135,206,250}  % 应用层 - 淡蓝色

\begin{document}
\begin{tikzpicture}[
    node distance=0.8cm and 1.5cm,
    layer/.style={rectangle, draw, thick, minimum width=14cm, minimum height=2.5cm, align=center, font=\small},
    module/.style={rectangle, draw, minimum width=3.5cm, minimum height=1.2cm, align=center, font=\footnotesize, rounded corners=3pt},
    arrow/.style={-Stealth, very thick},
    label/.style={font=\scriptsize, align=center}
]

% ==================== 第1层：数据管理层 ====================
\node[layer, fill=layer1!40] (L1) at (0,0) {};
\node[above=0.1cm of L1.north, font=\bfseries\large] {Layer 1: 数据管理层};

\node[module, fill=layer1!60] (mysql) at (-4, 0) {MySQL 8.0\\元数据库};
\node[module, fill=layer1!60, right=0.5cm of mysql] (redis) {Redis 7.0\\特征缓存};
\node[module, fill=layer1!60, right=0.5cm of redis] (minio) {MinIO\\视频存储};

\node[label, below=0.3cm of redis, align=center] {存储：视频45min≈450MB (H.265) | 特征70维×270片段≈2MB | TTL=7天};

% ==================== 第2层：特征提取层 ====================
\node[layer, fill=layer2!40, above=3cm of L1] (L2) {};
\node[above=0.1cm of L2.north, font=\bfseries\large] {Layer 2: 特征提取层 (Pipeline并行)};

\node[module, fill=layer2!70] (video_pipe) at (-4.2, 4.5) {
    \textbf{视频流水线}\\
    YOLOv8 → DeepSORT\\
    MediaPipe → ST-GCN\\
    \textcolor{red}{$F_v \in \mathbb{R}^{20}$}
};

\node[module, fill=layer2!70, right=0.3cm of video_pipe] (audio_pipe) {
    \textbf{音频流水线}\\
    Whisper → Wav2Vec2\\
    情感识别\\
    \textcolor{red}{$F_a \in \mathbb{R}^{15}$}
};

\node[module, fill=layer2!70, right=0.3cm of audio_pipe] (text_pipe) {
    \textbf{文本流水线}\\
    BERT → H-DAR\\
    NLP统计\\
    \textcolor{red}{$F_t \in \mathbb{R}^{35}$}
};

\node[label, below=0.3cm of audio_pipe, align=center] {
    耗时：视频0.82s | 音频0.37s | 文本0.15s | \textbf{总计0.82s/10s片段}
};

% 连接Layer 1 -> Layer 2
\draw[arrow] (minio.north) -- ++(0, 0.8) -| (video_pipe.south);
\draw[arrow] (minio.north) -- ++(0, 0.8) -| (audio_pipe.south);
\draw[arrow] (redis.north) -- ++(0, 0.8) -| (text_pipe.south);

% ==================== 第3层：模型推理层 ====================
\node[layer, fill=layer3!40, above=3cm of L2] (L3) {};
\node[above=0.1cm of L3.north, font=\bfseries\large] {Layer 3: 模型推理层};

\node[module, fill=layer3!70, minimum width=5cm, minimum height=1.5cm] (shape) at (-2.5, 9) {
    \textbf{SHAPE 融合模型}\\
    跨模态注意力 + BiLSTM\\
    342K参数 | 推理0.016s\\
    \textcolor{blue}{7类风格分类 (93.5\%)}
};

\node[module, fill=layer3!70, minimum width=5cm, minimum height=1.5cm, right=0.5cm of shape] (shap) {
    \textbf{SHAP 解释器}\\
    特征归因分析\\
    64背景样本\\
    \textcolor{blue}{可解释性分析}
};

\node[label, below=0.3cm of shape, align=center] {
    GPU加速 (TensorRT) | 批处理10x加速 | 注意力权重可视化
};

% 连接Layer 2 -> Layer 3
\draw[arrow] (video_pipe.north) -- ++(0, 0.8) -| (shape.south);
\draw[arrow] (audio_pipe.north) -- ++(0, 0.5) -- (shape.south);
\draw[arrow] (text_pipe.north) -- ++(0, 0.8) -| (shape.south);

% ==================== 第4层：应用服务层 ====================
\node[layer, fill=layer4!40, above=3cm of L3] (L4) {};
\node[above=0.1cm of L4.north, font=\bfseries\large] {Layer 4: 应用服务层};

\node[module, fill=layer4!70] (profile) at (-4.2, 13.5) {
    \textbf{画像生成}\\
    风格雷达图\\
    典型片段提取
};

\node[module, fill=layer4!70, right=0.3cm of profile] (viz) {
    \textbf{可视化}\\
    行为柱状图\\
    情绪曲线
};

\node[module, fill=layer4!70, right=0.3cm of viz] (analysis) {
    \textbf{分析服务}\\
    SMI相似度\\
    稳定性追踪
};

% 连接Layer 3 -> Layer 4
\draw[arrow] (shape.north) -- ++(0, 0.8) -| (profile.south);
\draw[arrow] (shape.north) -- ++(0, 0.5) -- (viz.south);
\draw[arrow] (shap.north) -- ++(0, 0.8) -| (analysis.south);

% ==================== 用户接口 ====================
\node[module, fill=layer4!90, above=1cm of viz, minimum width=8cm, minimum height=1cm] (ui) {
    \textbf{用户接口} (Vue.js + ECharts)\\
    教师端：风格画像查看 | 教研端：批量分析、跨教师对比
};

\draw[arrow] (profile.north) -- ++(0, 0.3) -| (ui.south);
\draw[arrow] (viz.north) -- (ui.south);
\draw[arrow] (analysis.north) -- ++(0, 0.3) -| (ui.south);

% ==================== 关键设计标注 ====================
\node[label, below=2.2cm of L1.south, draw, dashed, thick, minimum width=14cm, minimum height=2cm, align=left] (design) {
    \textbf{关键设计：}\\
    1. \textbf{异步任务队列} (Celery + RabbitMQ)：支持批量处理，失败重试\\
    2. \textbf{三级缓存策略}：L1=模型输出(Redis, 24h) | L2=特征向量(Redis, 7d) | L3=视频文件(MinIO, 永久)\\
    3. \textbf{水平扩展}：特征提取服务与模型推理服务可独立扩容，支持50并发/单机 → 200并发/分布式
};

% ==================== 整体标题 ====================
\node[above=0.5cm of ui, font=\Large\bfseries] {教师风格画像分析系统 - 四层架构设计};

\end{tikzpicture}
\end{document}
"""

    return tikz_code


def generate_multimodal_pipeline_tikz():
    """
    生成多模态特征提取Pipeline流程图
    对应 Algorithm 1
    """

    tikz_code = r"""
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{amsmath}

\usetikzlibrary{positioning, arrows.meta, shapes.geometric, calc}

% 定义颜色
\definecolor{video}{RGB}{255,127,127}
\definecolor{audio}{RGB}{127,255,127}
\definecolor{text}{RGB}{127,127,255}

\begin{document}
\begin{tikzpicture}[
    node distance=0.6cm and 1cm,
    process/.style={rectangle, draw, thick, minimum width=3cm, minimum height=0.8cm, align=center, font=\footnotesize, rounded corners=2pt},
    data/.style={rectangle, draw, dashed, minimum width=2.5cm, minimum height=0.6cm, align=center, font=\scriptsize},
    arrow/.style={-Stealth, thick},
    label/.style={font=\tiny, align=center}
]

% ==================== 视频流水线 ====================
\node[data, fill=video!20] (video_in) at (0,0) {视频片段\\10s@25fps};
\node[process, fill=video!30, below=of video_in] (extract) {ExtractFrames\\250帧};
\node[process, fill=video!40, below=of extract] (yolo) {YOLOv8-Batch\\人体检测};
\node[process, fill=video!50, below=of yolo] (deepsort) {DeepSORT\\教师追踪};
\node[process, fill=video!60, below=of deepsort] (mediapipe) {MediaPipe\\姿态估计};
\node[process, fill=video!70, below=of mediapipe] (stgcn) {ST-GCN\\动作识别};
\node[data, fill=video!80, below=of stgcn] (video_out) {$F_v \in \mathbb{R}^{20}$};

\draw[arrow] (video_in) -- (extract);
\draw[arrow] (extract) -- node[right, label] {0.05s} (yolo);
\draw[arrow] (yolo) -- node[right, label] {0.18s} (deepsort);
\draw[arrow] (deepsort) -- node[right, label] {0.12s} (mediapipe);
\draw[arrow] (mediapipe) -- node[right, label] {0.25s} (stgcn);
\draw[arrow] (stgcn) -- node[right, label] {0.18s} (video_out);

\node[above=0.3cm of video_in, font=\bfseries] {视频流水线 (0.82s)};

% ==================== 音频流水线 ====================
\node[data, fill=audio!20, right=4cm of video_in] (audio_in) {音频片段\\10s@16kHz};
\node[process, fill=audio!30, below=of audio_in] (load) {LoadAudio\\160k采样点};
\node[process, fill=audio!40, below=of load] (whisper) {Whisper\\语音转写};
\node[process, fill=audio!50, below=of whisper] (wav2vec) {Wav2Vec2\\声学嵌入};
\node[process, fill=audio!60, below=of wav2vec] (emotion) {Emotion\\情感分类};
\node[data, fill=audio!70, below=of emotion] (audio_out) {$F_a \in \mathbb{R}^{15}$};

\draw[arrow] (audio_in) -- (load);
\draw[arrow] (load) -- node[right, label] {0.05s} (whisper);
\draw[arrow] (whisper) -- node[right, label] {0.15s} (wav2vec);
\draw[arrow] (wav2vec) -- node[right, label] {0.08s} (emotion);
\draw[arrow] (emotion) -- node[right, label] {0.07s} (audio_out);

\node[above=0.3cm of audio_in, font=\bfseries] {音频流水线 (0.37s)};

% ==================== 文本流水线 ====================
\node[data, fill=text!20, right=4cm of audio_in] (text_in) {转写文本\\(await)};
\node[process, fill=text!30, below=of text_in] (bert) {BERT\\语义编码};
\node[process, fill=text!40, below=of bert] (hdar) {H-DAR\\意图识别};
\node[process, fill=text!50, below=of hdar] (nlp) {ComputeNLP\\统计特征};
\node[data, fill=text!60, below=of nlp] (text_out) {$F_t \in \mathbb{R}^{35}$};

\draw[arrow] (text_in) -- (bert);
\draw[arrow] (bert) -- node[right, label] {0.06s} (hdar);
\draw[arrow] (hdar) -- node[right, label] {0.04s} (nlp);
\draw[arrow] (nlp) -- node[right, label] {0.05s} (text_out);

% 依赖关系
\draw[arrow, dashed, color=gray] (whisper.east) -- ++(0.5, 0) |- (text_in.west);

\node[above=0.3cm of text_in, font=\bfseries] {文本流水线 (0.15s)};

% ==================== 并行合并 ====================
\node[process, fill=gray!30, below=3cm of audio_out, minimum width=8cm, minimum height=1cm] (merge) {
    \textbf{特征合并} \\
    $F = \{F_v, F_a, F_t\} \in \mathbb{R}^{70}$
};

\draw[arrow] (video_out) -| (merge);
\draw[arrow] (audio_out) -- (merge);
\draw[arrow] (text_out) -| (merge);

% ==================== 并行标注 ====================
\node[draw, dashed, thick, fit=(video_in)(video_out)(audio_in)(audio_out)(text_in)(text_out),
      label={[font=\bfseries]above:Pipeline 并行处理}] {};

% ==================== 整体标题 ====================
\node[above=1cm of video_in, font=\Large\bfseries] {多模态特征提取流程图 (Algorithm 1)};

\end{tikzpicture}
\end{document}
"""

    return tikz_code


if __name__ == "__main__":
    print("生成SHAPE网络架构图...")
    with open("shape_architecture.tex", "w", encoding="utf-8") as f:
        f.write(generate_shape_architecture_tikz())
    print("✓ 已生成 shape_architecture.tex")

    print("\n生成系统四层架构图...")
    with open("system_architecture.tex", "w", encoding="utf-8") as f:
        f.write(generate_system_architecture_tikz())
    print("✓ 已生成 system_architecture.tex")

    print("\n生成多模态特征提取流程图...")
    with open("multimodal_pipeline.tex", "w", encoding="utf-8") as f:
        f.write(generate_multimodal_pipeline_tikz())
    print("✓ 已生成 multimodal_pipeline.tex")

    print("\n" + "="*60)
    print("所有架构图生成完成！")
    print("="*60)
    print("\n使用方法：")
    print("1. 编译 LaTeX 文件生成 PDF：")
    print("   pdflatex shape_architecture.tex")
    print("   pdflatex system_architecture.tex")
    print("   pdflatex multimodal_pipeline.tex")
    print("\n2. 或转换为 PNG（需要 ImageMagick）：")
    print("   convert -density 300 shape_architecture.pdf -quality 90 shape_architecture.png")
    print("\n3. 插入论文的位置：")
    print("   - 图3.1: system_architecture.pdf (Section 3.1.2)")
    print("   - 图3.2: shape_architecture.pdf (Section 3.3.2)")
    print("   - 图5-1: multimodal_pipeline.pdf (Section 5.2.1, 可选)")
