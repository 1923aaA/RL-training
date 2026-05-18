import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
import os

# === 配置 ===
SAVE_DIR = "icons"
ICON_SIZE = 256
DPI = 100

# 科技风配色 (Clean Tech Light)
COLORS = {
    'S': '#00B69B',  # 雷达 - 青色
    'D': '#3D5AFE',  # 指挥 - 宝蓝
    'I': '#FF1744',  # 拦截 - 绯红
    'T': '#FF6D00'   # 目标 - 深橙
}

def setup_canvas():
    fig = plt.figure(figsize=(2, 2), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    return fig, ax

def save_icon(fig, name):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    path = os.path.join(SAVE_DIR, f"{name}.png")
    fig.savefig(path, transparent=True, dpi=DPI)
    plt.close(fig)
    print(f"[OK] Generated: {path}")

# ==========================================
# 1. 生成 I (Interceptor) - [战斗机形状]
# ==========================================
fig, ax = setup_canvas()
c = COLORS['I']

vertices = [
    (0, 1.0),    # 0. 机头 (MOVETO)
    (0.15, 0.4), # 1.
    (0.8, 0.0),  # 2. 右翼尖
    (0.8, -0.2), # 3.
    (0.2, -0.2), # 4.
    (0.3, -0.8), # 5. 右尾翼
    (0.1, -0.8), # 6.
    (0.0, -0.6), # 7. 尾喷口中心
    (-0.1, -0.8),# 8.
    (-0.3, -0.8),# 9. 左尾翼
    (-0.2, -0.2),# 10.
    (-0.8, -0.2),# 11.
    (-0.8, 0.0), # 12. 左翼尖
    (-0.15, 0.4),# 13.
    (0, 1.0)     # 14. 回到机头 (CLOSEPOLY)
]

# [修复] 顶点共15个，需要1个MOVETO, 13个LINETO, 1个CLOSEPOLY = 15个指令
codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 13 + [mpath.Path.CLOSEPOLY]

path = mpath.Path(vertices, codes)
patch = patches.PathPatch(path, facecolor=c, lw=0, alpha=1.0)
ax.add_patch(patch)

# 加一点细节线条
ax.plot([0, 0], [-0.6, 0.8], color='white', lw=1, alpha=0.5)

# 白色描边
patch_border = patches.PathPatch(path, facecolor='none', edgecolor='white', lw=3, zorder=-1)
ax.add_patch(patch_border)

save_icon(fig, "I")

# ==========================================
# 2. 生成 S (Sensor) - [雷达波形状]
# ==========================================
fig, ax = setup_canvas()
c = COLORS['S']

# 雷达底座
ax.add_patch(patches.Rectangle((-0.2, -0.9), 0.4, 0.3, color=c))
# 雷达锅 (弧形)
arc = patches.Arc((0, -0.3), 1.6, 1.2, angle=0, theta1=20, theta2=160, color=c, lw=5)
ax.add_patch(arc)
# 发射源
ax.add_patch(patches.Circle((0, -0.3), 0.15, color=c))
# 信号波 (3层)
for r in [0.5, 0.8, 1.1]:
    w = patches.Wedge((0, -0.3), r, 45, 135, width=0.08, color=c, alpha=0.7)
    ax.add_patch(w)

# 加白边
ax.add_patch(patches.Circle((0, -0.3), 1.15, color='white', fill=False, lw=0, zorder=-1))

save_icon(fig, "S")

# ==========================================
# 3. 生成 D (Decision) - [指挥中心/卫星]
# ==========================================
fig, ax = setup_canvas()
c = COLORS['D']

# 卫星本体
ax.add_patch(patches.Rectangle((-0.3, -0.3), 0.6, 0.6, color=c))
# 太阳能板 (左)
ax.add_patch(patches.Rectangle((-0.9, -0.2), 0.6, 0.4, color=c, alpha=0.8))
# 太阳能板 (右)
ax.add_patch(patches.Rectangle((0.3, -0.2), 0.6, 0.4, color=c, alpha=0.8))
# 信号线
ax.plot([0, 0], [0.3, 0.8], color=c, lw=2)
ax.add_patch(patches.Circle((0, 0.8), 0.1, color=c))
# 轨道环
circle = patches.Circle((0, 0), 0.7, fill=False, edgecolor=c, lw=1.5, linestyle='--')
ax.add_patch(circle)

# 描边
ax.add_patch(patches.Rectangle((-0.3, -0.3), 0.6, 0.6, fill=False, edgecolor='white', lw=2))

save_icon(fig, "D")

# ==========================================
# 4. 生成 T (Target) - [隐身无人机/敌机]
# ==========================================
fig, ax = setup_canvas()
c = COLORS['T'] # 橙色

# 飞翼布局 (B2幽灵风格)
vertices_t = [
    (0, 0.6),     # 机头 (MOVETO)
    (0.9, -0.4),  # 右翼尖
    (0.5, -0.6),
    (0.2, -0.4),
    (0, -0.6),    # 尾部中心
    (-0.2, -0.4),
    (-0.5, -0.6),
    (-0.9, -0.4), # 左翼尖
    (0, 0.6),     # 回到机头 (CLOSEPOLY)
]
# [修复] 顶点9个 = 1 + 7 + 1
codes_t = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 7 + [mpath.Path.CLOSEPOLY]

path_t = mpath.Path(vertices_t, codes_t)
patch_t = patches.PathPatch(path_t, facecolor=c, lw=0, alpha=1.0)
ax.add_patch(patch_t)

# 敌军标识 (菱形框)
diamond = patches.RegularPolygon((0, 0), numVertices=4, radius=1.0,
                                 orientation=0, fill=False, edgecolor=c, lw=2, linestyle='-')
ax.add_patch(diamond)

# 白边
patch_t_border = patches.PathPatch(path_t, facecolor='none', edgecolor='white', lw=2, zorder=-1)
ax.add_patch(patch_t_border)

save_icon(fig, "T")

print("------------------------------------------------")
print("成功! 战斗机、雷达、指挥中心、敌机图标已生成在 icons/ 文件夹中。")
print("请再次运行 render_kill_chain.bat 即可查看效果。")
print("------------------------------------------------")