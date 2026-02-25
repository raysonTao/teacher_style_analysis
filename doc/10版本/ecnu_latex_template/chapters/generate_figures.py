#!/usr/bin/env python3
"""
论文图表生成脚本
生成所有 matplotlib/seaborn 图像，供 LaTeX 论文使用。
中文字体说明：使用系统中文字体渲染中文标签。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
import os

# ── 中文字体配置 ──────────────────────────────────────────────────────────────
# 直接注册系统 Noto CJK 字体文件（matplotlib 的 .ttc 解析只读首个字族名）
_CJK_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
]
for _fp in _CJK_CANDIDATES:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)

# 查询 matplotlib 实际识别到的族名（通常为 'Noto Sans CJK JP'）
_cjk_names = [f.name for f in font_manager.fontManager.ttflist
              if 'Noto Sans CJK' in f.name]
_cjk_family = _cjk_names[0] if _cjk_names else 'DejaVu Sans'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [_cjk_family, 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
                        'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

BASE = os.path.dirname(os.path.abspath(__file__))
FIG3 = os.path.join(BASE, 'fig-3')
FIG4 = os.path.join(BASE, 'fig-4')
os.makedirs(FIG3, exist_ok=True)
os.makedirs(FIG4, exist_ok=True)

STYLES = ['理论讲授型', '耐心细致型', '启发引导型',
          '题目驱动型', '互动导向型', '逻辑推导型', '情感表达型']

# ── 配色方案 ──────────────────────────────────────────────────────────────────
BLUE   = '#2C6FBF'
LBLUE  = '#A8C8F0'
ORANGE = '#E87722'
GRAY   = '#888888'
LGRAY  = '#EEEEEE'
DGRAY  = '#333333'
RED    = '#C0392B'
GREEN  = '#27AE60'
BG     = '#F8F9FA'


# ══════════════════════════════════════════════════════════════════════════════
# 图 3.3 — 7×7 混淆矩阵
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig3_3():
    """SHAPE 模型 7 类风格分类混淆矩阵（占位示例数据）"""
    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False

    # 占位混淆矩阵（行=真实，列=预测），对角线主导
    raw = np.array([
        [42, 1, 2, 0, 1, 1, 1],   # 理论讲授型
        [ 1,28, 2, 1, 0, 1, 0],   # 耐心细致型
        [ 2, 1,38, 0, 2, 0, 1],   # 启发引导型
        [ 0, 1, 0,22, 1, 1, 0],   # 题目驱动型
        [ 1, 0, 2, 1,31, 0, 1],   # 互动导向型
        [ 1, 1, 0, 1, 0,27, 0],   # 逻辑推导型
        [ 1, 0, 1, 0, 1, 0,25],   # 情感表达型
    ], dtype=float)

    labels_short = ['理论\n讲授型', '耐心\n细致型', '启发\n引导型',
                    '题目\n驱动型', '互动\n导向型', '逻辑\n推导型', '情感\n表达型']

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(raw, cmap='Blues', aspect='auto',
                   vmin=0, vmax=raw.max())
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('样本数', fontsize=10)

    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(labels_short, fontsize=8.5)
    ax.set_yticklabels(labels_short, fontsize=8.5)
    ax.set_xlabel('预测类别', fontsize=11, labelpad=8)
    ax.set_ylabel('真实类别', fontsize=11, labelpad=8)

    thresh = raw.max() / 2.0
    for i in range(7):
        for j in range(7):
            color = 'white' if raw[i, j] > thresh else DGRAY
            ax.text(j, i, f'{int(raw[i,j])}',
                    ha='center', va='center', fontsize=9, color=color,
                    fontweight='bold' if i == j else 'normal')

    ax.set_title('注：括号内为占位示例数据，待实验后替换', fontsize=8,
                 color=GRAY, pad=4)
    fig.tight_layout()
    path = os.path.join(FIG3, 'fig-3-3.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图 4-2 — 课程级风格评分雷达图
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig4_2():
    """课程级风格评分向量——七边形雷达图（示例教师 A）"""
    scores = [0.78, 0.45, 0.62, 0.38, 0.55, 0.72, 0.31]
    N = len(STYLES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores_plot, color=BLUE, linewidth=2.2, linestyle='solid')
    ax.fill(angles, scores_plot, color=LBLUE, alpha=0.40)

    # 网格刻度
    ax.set_rlabel_position(30)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7.5, color=GRAY)
    ax.set_ylim(0, 1.05)

    ax.set_thetagrids(np.degrees(angles[:-1]), STYLES, fontsize=9.5)
    ax.tick_params(axis='x', pad=10)

    # 数值标注
    for angle, score in zip(angles[:-1], scores):
        ax.text(angle, score + 0.08, f'{score:.2f}', ha='center', va='center',
                fontsize=8.5, color=BLUE, fontweight='bold')

    ax.set_title('教师风格评分雷达图（示例：教师A，第3节课）',
                 fontsize=10, pad=20, color=DGRAY)
    ax.text(0.5, -0.06, '注：数值为示例占位数据，待实验后替换',
            transform=ax.transAxes, ha='center', fontsize=7.5, color=GRAY)

    path = os.path.join(FIG4, 'fig-4-2.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图 4-3 — SHAP 单次预测瀑布图
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig4_3():
    """SHAP 瀑布图：单一片段对"启发引导型"的预测贡献"""
    features = [
        ('问句比例(H-DAR)',        +0.312),
        ('walking 频率',           +0.187),
        ('语速变化(韵律)',          +0.145),
        ('gesturing 时长',         +0.118),
        ('情感: surprise',         +0.092),
        ('BERT 意图熵',             -0.076),
        ('standing 时长',          -0.091),
        ('情感: neutral',          -0.108),
        ('教学段落数',              -0.052),
        ('Wav2Vec 声学嵌入[3]',    +0.031),
    ]
    features.sort(key=lambda x: x[1])
    labels = [f[0] for f in features]
    values = [f[1] for f in features]

    base_val = 0.143   # E[f(X)] 基线

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [RED if v < 0 else BLUE for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor='white',
                   height=0.55, left=0)

    # 基线竖线
    ax.axvline(0, color=DGRAY, linewidth=0.8, linestyle='--', alpha=0.6)

    for bar, val in zip(bars, values):
        xpos = val + (0.005 if val >= 0 else -0.005)
        ha = 'left' if val >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center', ha=ha, fontsize=8.5, color=DGRAY)

    ax.set_xlabel('SHAP 贡献值', fontsize=10)
    ax.set_title(f'单片段预测"启发引导型"的 SHAP 瀑布图\n（基准值 = {base_val:.3f}，示例占位数据）',
                 fontsize=9.5, color=DGRAY)
    ax.set_xlim(-0.22, 0.40)

    pos_patch = mpatches.Patch(color=BLUE, label='正向贡献（提升概率）')
    neg_patch = mpatches.Patch(color=RED, label='负向贡献（降低概率）')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8.5, loc='lower right')

    fig.tight_layout()
    path = os.path.join(FIG4, 'fig-4-3.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图 4-4 — 典型片段自动提取展示
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig4_4():
    """3 类风格 × 3 片段的代表性片段展示网格（占位缩略图）"""
    selected_styles = ['启发引导型', '逻辑推导型', '互动导向型']
    confs = [
        [0.923, 0.891, 0.876],
        [0.912, 0.887, 0.864],
        [0.905, 0.878, 0.851],
    ]
    times = [
        ['06:12–06:34', '18:45–19:07', '32:01–32:22'],
        ['09:03–09:28', '21:14–21:38', '38:55–39:19'],
        ['12:40–13:01', '27:33–27:55', '41:18–41:40'],
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.25})
    cmap_list = [plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges]

    for row, (style, cmap) in enumerate(zip(selected_styles, cmap_list)):
        for col in range(3):
            ax = axes[row][col]
            # 占位缩略图：渐变色块模拟视频帧
            grad = np.linspace(0.3, 0.75, 60).reshape(6, 10)
            ax.imshow(grad, cmap=cmap, aspect='auto', interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])
            # 标注信息
            ax.set_title(f'片段 {col + 1}\n置信度: {confs[row][col]:.3f}\n时间: {times[row][col]}',
                         fontsize=7.5, pad=3, color=DGRAY)
            for spine in ax.spines.values():
                spine.set_edgecolor(BLUE)
                spine.set_linewidth(1.5)

        # 行标签
        axes[row][0].set_ylabel(style, fontsize=9.5, rotation=90,
                                labelpad=8, color=DGRAY, fontweight='bold')

    fig.suptitle('各风格置信度 Top-3 典型片段（占位示例）',
                 fontsize=11, color=DGRAY, y=1.01)
    path = os.path.join(FIG4, 'fig-4-4.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图 4-5 — 风格稳定性折线图
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig4_5():
    """跨 8 节课的风格评分时序追踪（3 主维度 + 趋势线）"""
    lessons = np.arange(1, 9)
    course_types = ['理论课', '习题课', '理论课', '探究课',
                    '习题课', '理论课', '探究课', '复习课']

    theory  = [0.78, 0.75, 0.80, 0.55, 0.72, 0.81, 0.58, 0.70]
    logic   = [0.72, 0.68, 0.75, 0.60, 0.65, 0.78, 0.62, 0.71]
    interact= [0.38, 0.42, 0.35, 0.65, 0.48, 0.33, 0.70, 0.45]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_facecolor(BG)

    def trend(x, y):
        z = np.polyfit(x, y, 1)
        return np.poly1d(z)(x)

    ax.plot(lessons, theory,   'o-', color=BLUE,   lw=2.2, ms=7, label='理论讲授型')
    ax.plot(lessons, trend(lessons, theory),   '--', color=BLUE,   lw=1.0, alpha=0.45)
    ax.plot(lessons, logic,    's-', color=ORANGE, lw=2.2, ms=7, label='逻辑推导型')
    ax.plot(lessons, trend(lessons, logic),    '--', color=ORANGE, lw=1.0, alpha=0.45)
    ax.plot(lessons, interact, '^-', color=GREEN,  lw=2.2, ms=7, label='互动导向型')
    ax.plot(lessons, trend(lessons, interact), '--', color=GREEN,  lw=1.0, alpha=0.45)

    ax.set_xticks(lessons)
    ax.set_xticklabels([f'第{i}课\n({ct})' for i, ct in zip(lessons, course_types)],
                       fontsize=8)
    ax.set_yticks(np.arange(0.2, 1.05, 0.1))
    ax.set_yticklabels([f'{v:.1f}' for v in np.arange(0.2, 1.05, 0.1)], fontsize=8.5)
    ax.set_ylim(0.20, 0.95)
    ax.set_xlabel('课次（课型）', fontsize=10, labelpad=6)
    ax.set_ylabel('风格评分', fontsize=10, labelpad=6)
    ax.set_title('教师 A 跨学期风格稳定性追踪（示例占位数据）',
                 fontsize=10.5, color=DGRAY, pad=8)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

    # 标注 σ 值
    for style, vals, color in [
        ('理论讲授', theory, BLUE),
        ('逻辑推导', logic, ORANGE),
        ('互动导向', interact, GREEN)
    ]:
        ax.annotate(f'σ={np.std(vals):.2f}',
                    xy=(8, vals[-1]), xytext=(8.15, vals[-1]),
                    fontsize=7.5, color=color)

    fig.tight_layout()
    path = os.path.join(FIG4, 'fig-4-5.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 通用 UI 原型图绘制工具
# ══════════════════════════════════════════════════════════════════════════════
NAV_COLOR  = '#2C3E50'
NAV_TXT    = '#ECF0F1'
CARD_BG    = '#FFFFFF'
CARD_BD    = '#DEE2E6'
BTN_BLUE   = '#3498DB'
BTN_GREEN  = '#27AE60'
BTN_ORANGE = '#E67E22'
FIELD_BG   = '#F1F3F5'
STATUS_COLORS = {
    '已完成': '#27AE60', '推理中': '#E67E22',
    '排队中': '#3498DB', '失败': '#E74C3C'
}


def draw_navbar(ax, fig_w, title='教师风格画像分析系统', username='教师 张老师'):
    """顶部导航栏"""
    ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, transform=ax.transAxes,
                                facecolor=NAV_COLOR, zorder=10))
    ax.text(0.02, 0.965, '≡', transform=ax.transAxes,
            fontsize=14, color=NAV_TXT, va='center', zorder=11)
    ax.text(0.06, 0.965, title, transform=ax.transAxes,
            fontsize=10.5, color=NAV_TXT, va='center', fontweight='bold', zorder=11)
    ax.text(0.88, 0.965, username, transform=ax.transAxes,
            fontsize=9, color=NAV_TXT, va='center', zorder=11)
    ax.text(0.96, 0.965, '⏻', transform=ax.transAxes,
            fontsize=11, color=NAV_TXT, va='center', zorder=11)


def draw_card(ax, x, y, w, h, title='', bg=CARD_BG, bd=CARD_BD,
              title_color=DGRAY, radius=0.01):
    """带标题的卡片区域"""
    card = FancyBboxPatch((x, y), w, h,
                          boxstyle=f'round,pad=0',
                          facecolor=bg, edgecolor=bd, linewidth=0.8,
                          transform=ax.transAxes, zorder=3)
    ax.add_patch(card)
    if title:
        ax.text(x + 0.015, y + h - 0.025, title, transform=ax.transAxes,
                fontsize=8.5, color=title_color, fontweight='bold', va='top', zorder=4)
    return card


def draw_button(ax, x, y, w, h, label, color=BTN_BLUE):
    """简单按钮"""
    btn = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.002',
                         facecolor=color, edgecolor='none',
                         transform=ax.transAxes, zorder=5)
    ax.add_patch(btn)
    ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes,
            fontsize=7.5, color='white', ha='center', va='center',
            fontweight='bold', zorder=6)


def draw_input_row(ax, x, y, w, label, value='', h=0.038):
    """表单输入行"""
    ax.text(x, y + h * 0.6, label, transform=ax.transAxes,
            fontsize=7.5, color=GRAY, va='center')
    field = FancyBboxPatch((x, y - h * 0.1), w, h * 0.8,
                           boxstyle='round,pad=0.002',
                           facecolor=FIELD_BG, edgecolor=CARD_BD, linewidth=0.6,
                           transform=ax.transAxes, zorder=4)
    ax.add_patch(field)
    if value:
        ax.text(x + 0.01, y + h * 0.25, value, transform=ax.transAxes,
                fontsize=7.5, color=DGRAY, va='center')


def placeholder_chart(ax, x, y, w, h, label='[图表区域]', color=LBLUE):
    """图表占位区域"""
    area = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0',
                          facecolor=color, edgecolor=CARD_BD, alpha=0.3, linewidth=0.6,
                          transform=ax.transAxes, zorder=4)
    ax.add_patch(area)
    ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes,
            fontsize=8, color=BLUE, ha='center', va='center',
            style='italic', zorder=5)


def new_ui_fig(title=''):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(LGRAY)
    fig.patch.set_facecolor(LGRAY)
    draw_navbar(ax, 12, title='教师风格画像分析系统')
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# 图 4-6 — 视频上传页面
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig4_6():
    fig, ax = new_ui_fig()

    # 页面标题
    ax.text(0.5, 0.88, '📤  上传课堂视频', transform=ax.transAxes,
            fontsize=13, color=DGRAY, ha='center', va='center', fontweight='bold')

    # 左侧：课程信息表单
    draw_card(ax, 0.03, 0.12, 0.38, 0.72, '课程基本信息', title_color=BLUE)
    fields = [
        ('教师姓名', '张明远老师'),
        ('课程名称', '高中物理·电磁感应'),
        ('授课日期', '2025-03-15'),
        ('课型', '理论课'),
        ('年级/班级', '高二(3)班'),
    ]
    for i, (label, val) in enumerate(fields):
        draw_input_row(ax, 0.06, 0.71 - i * 0.11, 0.30, label + '：', val)

    # 中央：拖拽上传区域
    draw_card(ax, 0.45, 0.35, 0.50, 0.49, '视频文件上传', title_color=BLUE)
    upload_area = FancyBboxPatch((0.47, 0.38), 0.46, 0.40,
                                 boxstyle='round,pad=0.01',
                                 facecolor='#F0F7FF', edgecolor=BLUE,
                                 linestyle='dashed', linewidth=1.5,
                                 transform=ax.transAxes, zorder=5)
    ax.add_patch(upload_area)
    ax.text(0.70, 0.60, '🎬', transform=ax.transAxes,
            fontsize=28, ha='center', va='center', color=BLUE, zorder=6)
    ax.text(0.70, 0.52, '将视频文件拖拽至此处', transform=ax.transAxes,
            fontsize=10, ha='center', va='center', color=BLUE, zorder=6)
    ax.text(0.70, 0.47, '支持 MP4 / MOV / AVI  最大 8 GB', transform=ax.transAxes,
            fontsize=8, ha='center', va='center', color=GRAY, zorder=6)
    draw_button(ax, 0.59, 0.39, 0.22, 0.045, '或 点击选择文件', color=BTN_BLUE)

    # 下方：进度条区域
    draw_card(ax, 0.45, 0.12, 0.50, 0.20, '上传进度', title_color=DGRAY)
    # 进度条
    ax.add_patch(plt.Rectangle((0.47, 0.18), 0.46, 0.025,
                                facecolor=FIELD_BG, edgecolor=CARD_BD,
                                linewidth=0.6, transform=ax.transAxes, zorder=4))
    ax.add_patch(plt.Rectangle((0.47, 0.18), 0.46 * 0.73, 0.025,
                                facecolor=BTN_BLUE, edgecolor='none',
                                transform=ax.transAxes, zorder=5))
    ax.text(0.70, 0.155, '正在上传… 73%  (865 MB / 1.18 GB)',
            transform=ax.transAxes, fontsize=8, ha='center', color=GRAY)

    # 底部按钮行
    draw_button(ax, 0.06, 0.05, 0.14, 0.048, '保存草稿', color=GRAY)
    draw_button(ax, 0.23, 0.05, 0.18, 0.048, '提交分析任务 ▶', color=BTN_GREEN)

    path = os.path.join(FIG4, 'fig-4-6.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图 4-7 — 任务管理页面
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig4_7():
    fig, ax = new_ui_fig()

    ax.text(0.5, 0.88, '📋  分析任务管理', transform=ax.transAxes,
            fontsize=13, color=DGRAY, ha='center', fontweight='bold')

    # 工具栏
    draw_button(ax, 0.03, 0.83, 0.12, 0.038, '+ 新建任务', BTN_BLUE)
    draw_button(ax, 0.16, 0.83, 0.10, 0.038, '🔄 刷新', GRAY)
    ax.text(0.80, 0.845, '状态筛选：', transform=ax.transAxes,
            fontsize=8.5, color=DGRAY, va='center')
    for i, (s, c) in enumerate([('全部', BLUE), ('进行中', BTN_ORANGE),
                                  ('已完成', BTN_GREEN)]):
        draw_button(ax, 0.88 + i * 0.035, 0.832, 0.032, 0.030, s, c)

    # 表格
    headers = ['任务ID', '教师/课程', '提交时间', '状态', '进度', '操作']
    col_x = [0.03, 0.12, 0.35, 0.56, 0.68, 0.84]
    col_w = [0.08, 0.22, 0.19, 0.11, 0.15, 0.14]

    # 表头
    ax.add_patch(plt.Rectangle((0.03, 0.768), 0.94, 0.048,
                                facecolor='#DDE8F5', edgecolor=CARD_BD,
                                linewidth=0.5, transform=ax.transAxes, zorder=3))
    for hdr, cx in zip(headers, col_x):
        ax.text(cx + 0.005, 0.792, hdr, transform=ax.transAxes,
                fontsize=8.5, color=DGRAY, fontweight='bold', va='center')

    # 数据行
    tasks = [
        ('T-2025031502', '张明远 / 电磁感应（高二）', '03-15 14:32', '推理中', '62%'),
        ('T-2025031501', '张明远 / 牛顿第二定律（高二）', '03-15 09:18', '已完成', '100%'),
        ('T-2025031403', '李晓燕 / 函数极值（高三）', '03-14 16:45', '已完成', '100%'),
        ('T-2025031402', '王大鹏 / 古诗词鉴赏（初三）', '03-14 11:20', '排队中', '0%'),
        ('T-2025031401', '李晓燕 / 数列求和（高三）', '03-14 08:55', '已完成', '100%'),
        ('T-2025031305', '王大鹏 / 文言文阅读（初三）', '03-13 15:40', '失败', '—'),
    ]

    for row_i, (tid, course, t, status, prog) in enumerate(tasks):
        row_y = 0.720 - row_i * 0.085
        bg = CARD_BG if row_i % 2 == 0 else '#F5F8FC'
        ax.add_patch(plt.Rectangle((0.03, row_y - 0.01), 0.94, 0.072,
                                    facecolor=bg, edgecolor=CARD_BD,
                                    linewidth=0.4, transform=ax.transAxes, zorder=3))
        vals = [tid, course, t, '', prog]
        for val, cx in zip(vals, col_x):
            ax.text(cx + 0.005, row_y + 0.025, val, transform=ax.transAxes,
                    fontsize=7.8, color=DGRAY, va='center')
        # 状态标签
        sc = STATUS_COLORS.get(status, GRAY)
        badge = FancyBboxPatch((col_x[3], row_y + 0.008), 0.09, 0.030,
                               boxstyle='round,pad=0.003',
                               facecolor=sc, edgecolor='none',
                               transform=ax.transAxes, zorder=4)
        ax.add_patch(badge)
        ax.text(col_x[3] + 0.045, row_y + 0.023, status,
                transform=ax.transAxes, fontsize=7.5, color='white',
                ha='center', va='center', zorder=5)
        # 操作按钮
        op_label = '查看报告' if status == '已完成' else ('取消' if status in ['排队中', '推理中'] else '重新提交')
        op_color = BTN_GREEN if status == '已完成' else (GRAY if status in ['排队中', '推理中'] else BTN_ORANGE)
        draw_button(ax, col_x[5], row_y + 0.010, 0.11, 0.028, op_label, op_color)

    path = os.path.join(FIG4, 'fig-4-7.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图 4-8 — 风格画像综合展示页面
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig4_8():
    fig, ax = new_ui_fig()

    # 课程信息栏
    ax.add_patch(plt.Rectangle((0.03, 0.84), 0.94, 0.06,
                                facecolor='#EBF4FF', edgecolor=CARD_BD,
                                linewidth=0.7, transform=ax.transAxes, zorder=3))
    ax.text(0.05, 0.872, '张明远老师  |  高中物理·电磁感应（高二3班）', transform=ax.transAxes,
            fontsize=9.5, color=DGRAY, va='center', fontweight='bold')
    badge = FancyBboxPatch((0.55, 0.853), 0.22, 0.030,
                           boxstyle='round,pad=0.003',
                           facecolor=BLUE, edgecolor='none',
                           transform=ax.transAxes, zorder=4)
    ax.add_patch(badge)
    ax.text(0.66, 0.868, '主导风格：逻辑推导型（置信度 87.3%）',
            transform=ax.transAxes, fontsize=8.5, color='white',
            ha='center', va='center', zorder=5)
    ax.text(0.82, 0.872, '分析完成：2025-03-15 15:42',
            transform=ax.transAxes, fontsize=8, color=GRAY, va='center')

    # 左侧：雷达图占位
    draw_card(ax, 0.03, 0.38, 0.55, 0.42, '教学风格雷达图', title_color=BLUE)
    placeholder_chart(ax, 0.05, 0.40, 0.50, 0.36, '七边形风格评分雷达图\n（逻辑推导型 0.78  ·  理论讲授型 0.72）', LBLUE)

    # 右侧：行为柱状图占位
    draw_card(ax, 0.61, 0.38, 0.36, 0.42, '行为分布统计', title_color=BLUE)
    placeholder_chart(ax, 0.63, 0.40, 0.32, 0.36,
                      '6 类动作频率与时长\n双轴柱状图', color='#FAE8C8')

    # 下半：Tab 面板
    draw_card(ax, 0.03, 0.08, 0.94, 0.27, '')
    for i, tab in enumerate(['📈 语音情绪时序曲线', '☁ 教学关键词云图']):
        tab_bg = '#EBF4FF' if i == 0 else CARD_BG
        ax.add_patch(plt.Rectangle((0.03 + i * 0.47, 0.31), 0.47, 0.030,
                                    facecolor=tab_bg, edgecolor=CARD_BD,
                                    linewidth=0.7, transform=ax.transAxes, zorder=3))
        ax.text(0.03 + i * 0.47 + 0.235, 0.325, tab,
                transform=ax.transAxes, fontsize=8.5, color=BLUE if i == 0 else GRAY,
                ha='center', va='center', fontweight='bold' if i == 0 else 'normal')

    placeholder_chart(ax, 0.05, 0.10, 0.90, 0.18,
                      '45 分钟课程情绪强度折线图（neutral / happy / surprise / ...）',
                      color='#E8F4E8')

    path = os.path.join(FIG4, 'fig-4-8.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图 4-9 — 可解释性与特征详情页面
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig4_9():
    fig, ax = new_ui_fig()

    ax.text(0.5, 0.88, '🔍  可解释性分析 — 张明远老师 · 电磁感应',
            transform=ax.transAxes, fontsize=11, color=DGRAY,
            ha='center', fontweight='bold')

    # 顶部：模态贡献饼图
    draw_card(ax, 0.03, 0.58, 0.30, 0.26, '模态权重贡献', title_color=BLUE)
    placeholder_chart(ax, 0.05, 0.60, 0.25, 0.21,
                      '三模态贡献饼图\n视频 38% · 文本 41% · 音频 21%', LBLUE)

    # 顶部中间：SHAP条形图
    draw_card(ax, 0.36, 0.58, 0.61, 0.26, 'Top-20 特征 SHAP 重要性（绝对值）', title_color=BLUE)
    placeholder_chart(ax, 0.38, 0.60, 0.56, 0.21,
                      '水平条形图（蓝=视频特征  橙=音频特征  绿=文本特征）',
                      color='#FFF3E0')

    # 中部：SHAP 散点图
    draw_card(ax, 0.03, 0.28, 0.94, 0.26, 'SHAP 特征分布散点图（Beeswarm）', title_color=BLUE)
    placeholder_chart(ax, 0.05, 0.30, 0.90, 0.21,
                      '各特征取值 vs SHAP贡献度（反映方向性影响）',
                      color='#F5F0FF')

    # 下部：典型片段
    draw_card(ax, 0.03, 0.08, 0.94, 0.17, '风格代表性片段回放', title_color=BLUE)
    for i, style in enumerate(['逻辑推导型', '理论讲授型', '互动导向型']):
        x = 0.05 + i * 0.32
        thumb = FancyBboxPatch((x, 0.10), 0.28, 0.095,
                               boxstyle='round,pad=0.003',
                               facecolor='#EEE', edgecolor=CARD_BD,
                               transform=ax.transAxes, zorder=4)
        ax.add_patch(thumb)
        ax.text(x + 0.14, 0.155, f'▶  {style}  置信度 {0.91 - i * 0.03:.2f}',
                transform=ax.transAxes, fontsize=8, ha='center',
                color=BLUE, va='center')
        ax.text(x + 0.14, 0.115, f'片段时间：{12 + i * 14}:03 – {12 + i * 14}:25',
                transform=ax.transAxes, fontsize=7, ha='center',
                color=GRAY, va='center')

    path = os.path.join(FIG4, 'fig-4-9.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图 4-10 — 风格演变追踪页面
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig4_10():
    fig, ax = new_ui_fig()

    ax.text(0.5, 0.88, '📈  风格演变追踪 — 张明远老师',
            transform=ax.transAxes, fontsize=12, color=DGRAY,
            ha='center', fontweight='bold')

    # 筛选工具栏
    draw_card(ax, 0.03, 0.80, 0.94, 0.05, '')
    ax.text(0.05, 0.826, '时间范围：', transform=ax.transAxes,
            fontsize=8.5, color=GRAY, va='center')
    for i, label in enumerate(['最近1月', '本学期', '自定义']):
        color = BLUE if i == 1 else FIELD_BG
        txt_color = 'white' if i == 1 else GRAY
        draw_button(ax, 0.16 + i * 0.08, 0.812, 0.07, 0.028, label, color)
    ax.text(0.42, 0.826, '课型筛选：', transform=ax.transAxes,
            fontsize=8.5, color=GRAY, va='center')
    for i, ct in enumerate(['全部', '理论课', '习题课', '探究课']):
        draw_button(ax, 0.52 + i * 0.07, 0.812, 0.06, 0.028, ct,
                    BLUE if i == 0 else FIELD_BG)
    draw_button(ax, 0.87, 0.812, 0.08, 0.028, '导出报告 ↓', BTN_GREEN)

    # 左侧：成长折线图
    draw_card(ax, 0.03, 0.20, 0.62, 0.56, '风格评分成长曲线（含趋势线）', title_color=BLUE)
    placeholder_chart(ax, 0.05, 0.22, 0.57, 0.50,
                      '多风格维度折线图\n含线性回归趋势线\n（第1课 → 第12课）',
                      LBLUE)

    # 右侧：稳定性热力图
    draw_card(ax, 0.68, 0.20, 0.29, 0.56, '风格稳定性热力图（σ 分布）', title_color=BLUE)
    placeholder_chart(ax, 0.70, 0.22, 0.24, 0.50,
                      '7 类风格 × 4 时段\n热力图\n（深色=稳定）',
                      color='#F0F0FF')

    # 底部摘要
    draw_card(ax, 0.03, 0.08, 0.94, 0.09, '阶段性摘要')
    ax.text(0.05, 0.137,
            '◉ 逻辑推导型（σ=0.06）高度稳定，为核心教学风格  '
            '◉ 互动导向型呈上升趋势（+0.18/学期）  '
            '◉ 情感表达型在探究课节显著高于理论课',
            transform=ax.transAxes, fontsize=8, color=DGRAY, va='center',
            wrap=True)

    path = os.path.join(FIG4, 'fig-4-10.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 图 4-11 — 批量分析与教研对比页面
# ══════════════════════════════════════════════════════════════════════════════
def gen_fig4_11():
    fig, ax = new_ui_fig()

    ax.text(0.5, 0.88, '📊  批量分析与教研对比（教研管理员视图）',
            transform=ax.transAxes, fontsize=12, color=DGRAY,
            ha='center', fontweight='bold')

    # 教师选择区
    draw_card(ax, 0.03, 0.55, 0.40, 0.29, '教师/课程选择', title_color=BLUE)
    teachers = [
        ('☑', '张明远', '物理', '12节', '已完成'),
        ('☑', '李晓燕', '数学', '8节', '已完成'),
        ('☑', '王大鹏', '语文', '10节', '已完成'),
        ('☐', '赵云飞', '英语', '6节', '分析中'),
    ]
    for i, (chk, name, subj, cnt, stat) in enumerate(teachers):
        y = 0.77 - i * 0.045
        ax.text(0.055, y, chk, transform=ax.transAxes, fontsize=10,
                color=BLUE if chk == '☑' else GRAY, va='center')
        ax.text(0.085, y, f'{name}  ({subj})  {cnt}', transform=ax.transAxes,
                fontsize=8, color=DGRAY, va='center')
        sc = BTN_GREEN if stat == '已完成' else BTN_ORANGE
        badge = FancyBboxPatch((0.33, y - 0.012), 0.07, 0.025,
                               boxstyle='round,pad=0.002',
                               facecolor=sc, edgecolor='none',
                               transform=ax.transAxes, zorder=4)
        ax.add_patch(badge)
        ax.text(0.365, y, stat, transform=ax.transAxes,
                fontsize=7.5, color='white', ha='center', va='center', zorder=5)

    draw_button(ax, 0.05, 0.57, 0.16, 0.032, '▶ 提交批量分析', BTN_GREEN)
    draw_button(ax, 0.22, 0.57, 0.11, 0.032, '⬇ 导出 Excel', BTN_BLUE)

    # 右侧：对比统计
    draw_card(ax, 0.46, 0.55, 0.51, 0.29, '教师群体风格统计', title_color=BLUE)
    placeholder_chart(ax, 0.48, 0.57, 0.47, 0.24,
                      '多教师风格均值对比柱状图\n（含误差棒 ±σ）',
                      color='#FFF3E0')

    # 对比雷达图
    draw_card(ax, 0.03, 0.13, 0.56, 0.38, '教师风格对比雷达图（多组叠加）', title_color=BLUE)
    placeholder_chart(ax, 0.05, 0.15, 0.52, 0.32,
                      '3 位教师风格雷达图叠加\n（张明远 / 李晓燕 / 王大鹏）',
                      LBLUE)

    # 差异分析
    draw_card(ax, 0.62, 0.13, 0.35, 0.38, '维度差异显著性分析', title_color=BLUE)
    rows = [
        ('逻辑推导型', '张 vs 李', 'p=0.023 *'),
        ('互动导向型', '张 vs 王', 'p=0.081'),
        ('情感表达型', '李 vs 王', 'p=0.004 **'),
        ('启发引导型', '全体',    'p=0.156'),
    ]
    ax.text(0.64, 0.465, '风格维度          教师对比       显著性',
            transform=ax.transAxes, fontsize=7.5, color=GRAY,
            va='center', fontweight='bold')
    for i, (dim, pair, sig) in enumerate(rows):
        y = 0.430 - i * 0.058
        ax.text(0.64, y, f'{dim}    {pair}    {sig}',
                transform=ax.transAxes, fontsize=7.5, color=DGRAY, va='center')
        ax.plot([0.62, 0.97], [y - 0.010, y - 0.010],
                color=CARD_BD, linewidth=0.5, transform=ax.transAxes)

    path = os.path.join(FIG4, 'fig-4-11.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=== 开始生成论文图像 ===\n')
    gen_fig3_3()
    gen_fig4_2()
    gen_fig4_3()
    gen_fig4_4()
    gen_fig4_5()
    gen_fig4_6()
    gen_fig4_7()
    gen_fig4_8()
    gen_fig4_9()
    gen_fig4_10()
    gen_fig4_11()
    print('\n=== 全部完成 ===')
