import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 定义框的样式
def draw_box(ax, x, y, width, height, text, fontsize=16, boxstyle="round,pad=0.3"):
    """绘制圆角矩形框"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle=boxstyle,
        linewidth=2.5,
        edgecolor='black',
        facecolor='white',
        zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            weight='bold', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, width=0.08):
    """绘制箭头"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        linewidth=2.5,
        color='black',
        zorder=1
    )
    ax.add_patch(arrow)

# 绘制章节框
# 第一章
draw_box(ax, 7, 9, 4, 0.8, '第一章 绪论')

# 第二章  
draw_box(ax, 7, 7.5, 5, 0.8, '第二章 理论基础与相关研究')

# 第三章和第四章 (并列)
draw_box(ax, 3.5, 5.5, 4.5, 1, '第三章 研究方法与总体设计', fontsize=15)
draw_box(ax, 10.5, 5.5, 4.5, 1, '第四章 多模态特征提取', fontsize=15)

# 第五章
draw_box(ax, 7, 3.5, 5, 0.8, '第五章 教师风格画像分析系统设计与实现', fontsize=14)

# 第六章
draw_box(ax, 7, 1.5, 4, 0.8, '第六章 总结与展望')

# 绘制箭头
# 第一章 → 第二章
draw_arrow(ax, 7, 8.6, 7, 7.9)

# 第二章 → 第三章和第四章
draw_arrow(ax, 7, 7.1, 7, 6.5)
draw_arrow(ax, 5, 6.5, 3.5, 6)
draw_arrow(ax, 9, 6.5, 10.5, 6)

# 第三章和第四章 → 第五章
draw_arrow(ax, 3.5, 5, 3.5, 4.5)
draw_arrow(ax, 10.5, 5, 10.5, 4.5)
draw_arrow(ax, 3.5, 4.5, 7, 3.9)
draw_arrow(ax, 10.5, 4.5, 7, 3.9)

# 第五章 → 第六章
draw_arrow(ax, 7, 3.1, 7, 1.9)

# 绘制虚线框和标注
# 理论与实验部分的虚线框
dashed_box = mpatches.FancyBboxPatch(
    (0.5, 4.7), 13, 2.5,
    boxstyle="round,pad=0.2",
    linewidth=2,
    edgecolor='gray',
    facecolor='none',
    linestyle='--',
    zorder=0
)
ax.add_patch(dashed_box)

# 左侧标注
ax.text(0.3, 6, '理论与实验', ha='center', va='center', 
        fontsize=14, style='italic', color='gray', rotation=90)

# 右侧标注 - 第五章的内容说明
ax.text(12.8, 3.5, '系统设计与实现', ha='left', va='center',
        fontsize=12, style='italic', color='gray')
ax.plot([12.5, 12.7], [3.5, 3.5], 'k-', linewidth=1.5)

# 添加副标题说明（在虚线框内）
ax.text(7, 4.3, '方法设计 + 特征提取 + 融合建模', ha='center', va='center',
        fontsize=11, style='italic', color='#555555', 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                  edgecolor='none', alpha=0.6))

plt.tight_layout()
plt.savefig('论文组织结构_new.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ 结构图已生成: 论文组织结构_new.png")
