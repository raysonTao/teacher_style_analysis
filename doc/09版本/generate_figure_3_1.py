#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成图3.1：SHAPE引擎四层架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 设置中文字体 - 使用系统的Noto Sans CJK SC
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 创建图形
fig, ax = plt.subplots(figsize=(14, 16))
ax.set_xlim(0, 14)
ax.set_ylim(0, 16)
ax.axis('off')

# 颜色方案
color_layer1 = '#E3F2FD'  # 浅蓝色
color_layer2 = '#FFF9C4'  # 浅黄色
color_layer3 = '#F8BBD0'  # 浅粉色
color_layer4 = '#C8E6C9'  # 浅绿色
color_innovation = '#FFE082'  # 创新点标记颜色
color_arrow = '#424242'

# ========== 标题 ==========
ax.text(7, 15.5, 'SHAPE引擎四层架构', fontsize=20, weight='bold', ha='center')
ax.text(7, 15, '(Semantic Hierarchical Attention Profiling Engine)',
        fontsize=11, ha='center', style='italic', color='gray')

# ========== 第一层：数据预处理层 ==========
layer1_y = 13.5
layer1_box = FancyBboxPatch((0.5, layer1_y-1.8), 13, 2,
                            boxstyle="round,pad=0.1",
                            edgecolor='#1976D2', facecolor=color_layer1, linewidth=2)
ax.add_patch(layer1_box)

ax.text(7, layer1_y+0.1, '第一层：数据预处理层 (Data Preprocessing Layer)',
        fontsize=13, weight='bold', ha='center')

# 输入
ax.text(2, layer1_y-0.5, '输入：课堂录像\n(Video + Audio)',
        fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# 数据同步
ax.text(5, layer1_y-0.5, '音视频同步\n(Cross-Correlation)',
        fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# 创新点1：语义驱动分段
innovation1_box = FancyBboxPatch((7.2, layer1_y-0.85), 3.6, 0.7,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='#FF6F00', facecolor=color_innovation, linewidth=2)
ax.add_patch(innovation1_box)
ax.text(9, layer1_y-0.5, '⭐创新点1：语义驱动分段',
        fontsize=10, weight='bold', ha='center', color='#D84315')
ax.text(9, layer1_y-0.75, '(完整率 76.6%→95.3%)',
        fontsize=8, ha='center', style='italic')

# 输出
ax.text(12, layer1_y-0.5, '输出：N≈175个\n语义单元/课',
        fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# ========== 第二层：特征提取层 ==========
layer2_y = 10.5
layer2_box = FancyBboxPatch((0.5, layer2_y-3.2), 13, 3.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='#F57C00', facecolor=color_layer2, linewidth=2)
ax.add_patch(layer2_box)

ax.text(7, layer2_y+0.2, '第二层：特征提取层 (Feature Extraction Layer)',
        fontsize=13, weight='bold', ha='center')

# 三个模态并列
# 视觉模态
visual_x = 2.3
ax.text(visual_x, layer2_y-0.5, '视觉模态 (20维)', fontsize=10, weight='bold', ha='center')
ax.text(visual_x, layer2_y-0.9, 'YOLOv8检测', fontsize=8, ha='center')
ax.add_patch(FancyBboxPatch((visual_x-0.4, layer2_y-1.15), 0.8, 0.15,
                            facecolor='#BBDEFB', edgecolor='gray'))
ax.text(visual_x, layer2_y-1.35, '↓', fontsize=12, ha='center')
ax.text(visual_x, layer2_y-1.55, 'DeepSORT追踪', fontsize=8, ha='center')
ax.add_patch(FancyBboxPatch((visual_x-0.4, layer2_y-1.8), 0.8, 0.15,
                            facecolor='#BBDEFB', edgecolor='gray'))
ax.text(visual_x, layer2_y-2.0, '↓', fontsize=12, ha='center')
ax.text(visual_x, layer2_y-2.2, 'MediaPipe骨骼', fontsize=8, ha='center')
ax.add_patch(FancyBboxPatch((visual_x-0.4, layer2_y-2.45), 0.8, 0.15,
                            facecolor='#BBDEFB', edgecolor='gray'))
ax.text(visual_x, layer2_y-2.65, '↓', fontsize=12, ha='center')
ax.text(visual_x, layer2_y-2.85, 'ST-GCN建模', fontsize=8, ha='center', weight='bold')
ax.add_patch(FancyBboxPatch((visual_x-0.4, layer2_y-3.1), 0.8, 0.15,
                            facecolor='#FFE082', edgecolor='gray'))

# 音频模态
audio_x = 5.5
ax.text(audio_x, layer2_y-0.5, '音频模态 (15维)', fontsize=10, weight='bold', ha='center')
ax.text(audio_x, layer2_y-0.9, 'Wav2Vec 2.0', fontsize=8, ha='center', weight='bold')
ax.add_patch(FancyBboxPatch((audio_x-0.5, layer2_y-1.15), 1.0, 0.15,
                            facecolor='#FFE082', edgecolor='gray'))
ax.text(audio_x, layer2_y-1.35, '深度声学表征', fontsize=8, ha='center')
ax.text(audio_x, layer2_y-1.65, '↓', fontsize=12, ha='center')
ax.text(audio_x, layer2_y-1.85, '情感分类头', fontsize=8, ha='center')
ax.add_patch(FancyBboxPatch((audio_x-0.4, layer2_y-2.1), 0.8, 0.15,
                            facecolor='#FFCCBC', edgecolor='gray'))
ax.text(audio_x, layer2_y-2.5, '10维统计特征\n+5维情感特征', fontsize=7, ha='center')

# 文本模态（包含创新点2）
text_x = 9.5
innovation2_box = FancyBboxPatch((text_x-1.1, layer2_y-0.7), 2.2, 0.4,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='#FF6F00', facecolor=color_innovation, linewidth=2)
ax.add_patch(innovation2_box)
ax.text(text_x, layer2_y-0.5, '⭐创新点2：文本模态 (35维)', fontsize=10, weight='bold', ha='center', color='#D84315')

ax.text(text_x, layer2_y-0.9, 'Whisper ASR', fontsize=8, ha='center')
ax.add_patch(FancyBboxPatch((text_x-0.4, layer2_y-1.15), 0.8, 0.15,
                            facecolor='#BBDEFB', edgecolor='gray'))
ax.text(text_x, layer2_y-1.35, '↓', fontsize=12, ha='center')
ax.text(text_x, layer2_y-1.55, 'BERT编码', fontsize=8, ha='center')
ax.add_patch(FancyBboxPatch((text_x-0.4, layer2_y-1.8), 0.8, 0.15,
                            facecolor='#BBDEFB', edgecolor='gray'))
ax.text(text_x, layer2_y-2.0, '↓', fontsize=12, ha='center')
ax.text(text_x, layer2_y-2.2, 'H-DAR (10类)', fontsize=8, ha='center', weight='bold')
ax.add_patch(FancyBboxPatch((text_x-0.5, layer2_y-2.45), 1.0, 0.15,
                            facecolor='#FFE082', edgecolor='gray'))
ax.text(text_x, layer2_y-2.8, '层次化意图识别\n(F1: 0.70→0.89)', fontsize=7, ha='center', style='italic')

# 右侧性能提升
ax.text(12, layer2_y-1, 'vs MFCC: +6.4%\nvs 单帧: +17.7%\nvs 关键词: +12.6%',
        fontsize=7, ha='left', bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='green'))

# ========== 第三层：融合分类层 ==========
layer3_y = 5.5
layer3_box = FancyBboxPatch((0.5, layer3_y-2.5), 13, 3,
                            boxstyle="round,pad=0.1",
                            edgecolor='#C2185B', facecolor=color_layer3, linewidth=2)
ax.add_patch(layer3_box)

innovation3_box = FancyBboxPatch((2.5, layer3_y+0.35), 9, 0.4,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='#FF6F00', facecolor=color_innovation, linewidth=2)
ax.add_patch(innovation3_box)
ax.text(7, layer3_y+0.55, '⭐创新点3：第三层：SHAPE跨模态注意力融合 (Fusion & Classification Layer)',
        fontsize=11, weight='bold', ha='center', color='#D84315')

# SHAPE五模块
modules_y = layer3_y-0.2
ax.text(2, modules_y, '模块1:\n特征投影', fontsize=8, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
ax.text(2, modules_y-0.35, '→512维', fontsize=7, ha='center', style='italic')

ax.text(4, modules_y, '模块2:\n跨模态注意力', fontsize=8, ha='center', weight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFE082', edgecolor='#FF6F00', linewidth=2))
ax.text(4, modules_y-0.35, '6个α权重', fontsize=7, ha='center', style='italic')

ax.text(6.5, modules_y, '模块3:\nBiLSTM', fontsize=8, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
ax.text(6.5, modules_y-0.35, '时序建模', fontsize=7, ha='center', style='italic')

ax.text(9, modules_y, '模块4:\n注意力池化', fontsize=8, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
ax.text(9, modules_y-0.35, 'β权重', fontsize=7, ha='center', style='italic')

ax.text(11.5, modules_y, '模块5:\n风格分类器', fontsize=8, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
ax.text(11.5, modules_y-0.35, '7类输出', fontsize=7, ha='center', style='italic')

# 箭头连接
for i, x in enumerate([2, 4, 6.5, 9]):
    arrow = FancyArrowPatch((x+0.5, modules_y), (x+1, modules_y),
                           arrowstyle='->', mutation_scale=15, linewidth=2, color=color_arrow)
    ax.add_patch(arrow)

# 性能对比
ax.text(7, layer3_y-1.5, '性能提升：', fontsize=9, ha='center', weight='bold')
ax.text(7, layer3_y-1.8, 'vs Early Fusion: +6.2%  |  vs Late Fusion: +3.8%',
        fontsize=8, ha='center')
ax.text(7, layer3_y-2.1, 'vs 最佳单模态: +13.1% (78.3%→91.4%)',
        fontsize=8, ha='center', style='italic', color='#1B5E20')

# ========== 第四层：画像生成层 ==========
layer4_y = 2
layer4_box = FancyBboxPatch((0.5, layer4_y-1.5), 13, 2,
                            boxstyle="round,pad=0.1",
                            edgecolor='#388E3C', facecolor=color_layer4, linewidth=2)
ax.add_patch(layer4_box)

ax.text(7, layer4_y+0.4, '第四层：画像生成层 (Profiling & Application Layer)',
        fontsize=13, weight='bold', ha='center')

# 三大输出
output_y = layer4_y-0.3
ax.text(2.5, output_y, '风格分类结果', fontsize=9, ha='center', weight='bold')
ax.text(2.5, output_y-0.35, '主导风格+置信度\nTop-2覆盖率98.1%',
        fontsize=7, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

ax.text(7, output_y, '模态贡献度分析', fontsize=9, ha='center', weight='bold')
ax.text(7, output_y-0.35, '基于α权重\n情感型:音频0.62\n互动型:视觉0.50',
        fontsize=7, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

ax.text(11.5, output_y, '典型片段提取', fontsize=9, ha='center', weight='bold')
ax.text(11.5, output_y-0.35, '基于β权重\nTop-K关键时刻\n可解释性分析',
        fontsize=7, ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# 最终性能指标
ax.text(7, layer4_y-1.2, '最终性能：准确率93.5% | F1=0.91 | Kappa=0.89',
        fontsize=10, ha='center', weight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2))

# ========== 层间箭头 ==========
# 第1层→第2层
arrow1 = FancyArrowPatch((7, layer1_y-1.9), (7, layer2_y+0.5),
                        arrowstyle='->', mutation_scale=30, linewidth=3, color=color_arrow)
ax.add_patch(arrow1)

# 第2层→第3层
arrow2 = FancyArrowPatch((7, layer2_y-3.3), (7, layer3_y+0.8),
                        arrowstyle='->', mutation_scale=30, linewidth=3, color=color_arrow)
ax.add_patch(arrow2)

# 第3层→第4层
arrow3 = FancyArrowPatch((7, layer3_y-2.6), (7, layer4_y+0.6),
                        arrowstyle='->', mutation_scale=30, linewidth=3, color=color_arrow)
ax.add_patch(arrow3)

# ========== 图例 ==========
legend_elements = [
    mpatches.Patch(facecolor=color_innovation, edgecolor='#FF6F00', linewidth=2, label='核心创新点'),
    mpatches.Patch(facecolor='#FFE082', edgecolor='gray', label='深度学习技术'),
    mpatches.Patch(facecolor='white', edgecolor='gray', label='传统处理'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

# ========== 底部注释 ==========
ax.text(7, 0.3, '注：三项创新形成完整改进链，语义分段→H-DAR→SHAPE，最终准确率93.5%',
        fontsize=8, ha='center', style='italic', color='gray')

plt.tight_layout()
plt.savefig('/home/rayson/code/teacher_style_analysis/doc/09版本/图3.1_SHAPE引擎四层架构图.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/rayson/code/teacher_style_analysis/doc/09版本/图3.1_SHAPE引擎四层架构图.pdf',
            bbox_inches='tight', facecolor='white')
print("✅ 图3.1已生成：")
print("   - PNG格式: doc/09版本/图3.1_SHAPE引擎四层架构图.png (300 DPI)")
print("   - PDF格式: doc/09版本/图3.1_SHAPE引擎四层架构图.pdf (矢量图)")
