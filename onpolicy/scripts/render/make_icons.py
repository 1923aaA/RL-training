import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# === 配置 ===
SAVE_DIR = "icons"
ICON_SIZE = 256  # 输出图片分辨率
DPI = 100

# 风格配色 (对应 Clean Tech Light)
COLORS = {
    'S': '#00B69B',  # Emerald Green
    'D': '#3D5AFE',  # Royal Blue
    'I': '#FF1744',  # Crimson Red
    'T': '#FF6D00'  # Deep Orange
}


def setup_canvas():
    """创建一个透明背景的画布"""
    fig = plt.figure(figsize=(2, 2), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    return fig, ax


def save_icon(fig, name):
    """保存图标"""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    path = os.path.join(SAVE_DIR, f"{name}.png")
    fig.savefig(path, transparent=True, dpi=DPI)
    plt.close(fig)
    print(f"Generated: {path}")


# ============================
# 1. 生成 S (Sensor) - 雷达波
# ============================
fig, ax = setup_canvas()
c = COLORS['S']
# 外圈
ax.add_patch(patches.Circle((0, 0), 1.0, color=c, alpha=0.2))
ax.add_patch(patches.Circle((0, 0), 1.0, color=c, fill=False, lw=4))
# 内圈
ax.add_patch(patches.Circle((0, 0), 0.6, color=c, fill=False, lw=3))
# 核心点
ax.add_patch(patches.Circle((0, 0), 0.2, color=c))
# 扫描线 (扇形)
wedge = patches.Wedge((0, 0), 1.0, 30, 80, color=c, alpha=0.6)
ax.add_patch(wedge)
# 加个白边增强对比度
ax.add_patch(patches.Circle((0, 0), 1.02, color='white', fill=False, lw=2, zorder=-1))

save_icon(fig, "S")

# ============================
# 2. 生成 D (Decision) - 指挥中心
# ============================
fig, ax = setup_canvas()
c = COLORS['D']
# 六边形底座
hexagon = patches.RegularPolygon((0, 0), numVertices=6, radius=1.0,
                                 orientation=np.pi / 2, facecolor=c, alpha=0.2, edgecolor=c, linewidth=4)
ax.add_patch(hexagon)
# 内部方块 (象征芯片/计算)
rect = patches.Rectangle((-0.4, -0.4), 0.8, 0.8, facecolor=c, alpha=0.9)
ax.add_patch(rect)
# 连接线
ax.plot([0, 0], [-1, -0.4], color=c, lw=3)
ax.plot([0, 0], [0.4, 1], color=c, lw=3)
ax.plot([-0.86, -0.4], [0, 0], color=c, lw=3)
ax.plot([0.4, 0.86], [0, 0], color=c, lw=3)
# 白边
border = patches.RegularPolygon((0, 0), numVertices=6, radius=1.02,
                                orientation=np.pi / 2, facecolor='none', edgecolor='white', linewidth=2, zorder=-1)
ax.add_patch(border)

save_icon(fig, "D")

# ============================
# 3. 生成 I (Interceptor) - 战斗机/导弹
# ============================
fig, ax = setup_canvas()
c = COLORS['I']
# 绘制类似战斗机的三角箭头
path_x = [0, -0.7, 0, 0.7, 0]
path_y = [1.0, -0.8, -0.4, -0.8, 1.0]
ax.fill(path_x, path_y, color=c, alpha=0.9)
# 边框
ax.plot(path_x, path_y, color='white', lw=3, zorder=-1)  # 白底边
ax.plot(path_x, path_y, color=c, lw=2)

save_icon(fig, "I")

# ============================
# 4. 生成 T (Target) - 敌机 (菱形)
# ============================
fig, ax = setup_canvas()
c = COLORS['T']  # 深橙色
# 菱形 (NATO标准敌军符号)
diamond = patches.RegularPolygon((0, 0), numVertices=4, radius=1.0,
                                 orientation=0, facecolor=c, alpha=0.8)
ax.add_patch(diamond)
# 内部加个叉
ax.plot([-0.5, 0.5], [-0.5, 0.5], color='white', lw=3)
ax.plot([-0.5, 0.5], [0.5, -0.5], color='white', lw=3)
# 外白边
border = patches.RegularPolygon((0, 0), numVertices=4, radius=1.02,
                                orientation=0, facecolor='none', edgecolor='white', linewidth=2, zorder=-1)
ax.add_patch(border)

save_icon(fig, "T")

print("All icons generated in 'icons/' folder!")