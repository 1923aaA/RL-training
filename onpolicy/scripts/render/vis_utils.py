import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D, proj3d
import numpy as np
import warnings
import os
import copy
from collections import deque
# 忽略 matplotlib 的警告
warnings.filterwarnings("ignore")


class KillChainVisualizer:
    def __init__(self, env, space_size=100.0, refresh_interval=0.1, save_frames=False):
        self.env = env
        self.refresh_interval = refresh_interval

        self.save_frames = save_frames  # <--- [新增] 保存开关状态
        self.save_dir = "captures"
        if self.save_frames and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"[Visualizer] 全帧录制模式开启，保存路径: {os.path.abspath(self.save_dir)}")

        # === [参数配置] ===
        self.sx = getattr(env.cfg, 'space_size_x', space_size)
        self.sy = getattr(env.cfg, 'space_size_y', space_size)
        self.sz = getattr(env.cfg, 'space_size_z', space_size)

        self.max_span = max(self.sx, self.sy, self.sz)
        self.label_offset = self.sz * 0.05

        self.gx = getattr(env.cfg, 'grid_size_x', 10)
        self.gy = getattr(env.cfg, 'grid_size_y', 10)
        self.gz = getattr(env.cfg, 'grid_size_z', 10)

        # 加载图标
        self.icon_cache = {}
        self.use_icons = self._load_custom_icons()

        plt.ion()

        # === [风格: 保持您上传文件的浅色风格] ===
        self.bg_color = '#F5F7FA'
        self.sidebar_bg = '#FFFFFF'
        self.accent_color = '#2979FF'
        self.grid_color = '#CFD8DC'
        self.text_primary = '#263238'
        self.text_secondary = '#78909C'
        self.divider_color = '#F0F0F0'

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.facecolor'] = self.bg_color
        plt.rcParams['figure.facecolor'] = self.bg_color
        plt.rcParams['text.color'] = self.text_primary

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("TACTICAL DASHBOARD")

        # 左侧 UI (28%)
        self.ax_ui = self.fig.add_axes([0.0, 0.0, 0.28, 1.0])
        self.ax_ui.axis('off')
        self.ax_ui.add_patch(
            patches.Rectangle((0, 0), 1, 1, facecolor=self.sidebar_bg, transform=self.ax_ui.transAxes, zorder=-1))
        self.ax_ui.plot([1, 1], [0, 1], color='#E0E0E0', lw=2)

        # 右侧 3D (72%)
        self.ax = self.fig.add_axes([0.28, 0.05, 0.72, 0.95], projection='3d')
        self.ax.view_init(elev=28, azim=-55)
        self.ax.set_axis_off()

        # 轨迹
        self.trails = {i: deque(maxlen=2) for i in range(env.n_agents)}

        # 预计算圆圈
        theta = np.linspace(0, 2 * np.pi, 60)
        self.circle_x = np.cos(theta)
        self.circle_y = np.sin(theta)

        # === [配色表] ===
        self.colors = {
            'S': '#00B69B', 'D': '#3D5AFE', 'I': '#FF1744',
            'T_PENDING': '#90A4AE', 'T_DETECTED': '#FF6D00',
            'T_ASSIGNED': '#E65100', 'T_STRIKING': '#6200EA', 'T_COMPLETED': '#212121'
        }
        self.chain_history = {}  # 格式: {task_id: {'pts': points, 'life': 3}}
        self.finished_chains = set()  # [新增] 已经展示结束的任务黑名单

    def _load_custom_icons(self):
        current_dir = os.path.dirname(__file__)
        icon_dir = os.path.join(current_dir, 'icons')
        mapping = {'S': 'S.png', 'D': 'D.png', 'I': 'I.png', 'TARGET': 'T.png'}
        loaded_any = False
        if not os.path.exists(icon_dir): return False
        for key, filename in mapping.items():
            path = os.path.join(icon_dir, filename)
            if os.path.exists(path):
                try:
                    img = plt.imread(path)
                    self.icon_cache[key] = img
                    loaded_any = True
                except:
                    pass
        return loaded_any

    def _get_target_color(self, stage_name):
        mapping = {
            'PENDING': self.colors['T_PENDING'], 'DETECTED': self.colors['T_DETECTED'],
            'ASSIGNED': self.colors['T_ASSIGNED'], 'STRIKING': self.colors['T_STRIKING'],
            'COMPLETED': self.colors['T_COMPLETED'], 'FAILED': '#C51162'
        }
        return mapping.get(stage_name, '#000000')

    def _draw_range_circle(self, x, y, z, radius, color):
        if radius <= 0: return
        cx = x + radius * self.circle_x
        cy = y + radius * self.circle_y
        cz = np.full_like(cx, z)
        self.ax.plot(cx, cy, cz, color=color, alpha=0.5, linestyle='--', linewidth=0.8)

    def _draw_ui_divider(self, y_pos):
        self.ax_ui.plot([0.08, 0.92], [y_pos, y_pos], color=self.divider_color, lw=1.5)

    def _draw_icon_marker(self, x, y, z, key, color, zoom=0.08, alpha=1.0):
        if key in self.icon_cache:
            imagebox = OffsetImage(self.icon_cache[key], zoom=zoom, alpha=alpha)
            x2, y2, _ = proj3d.proj_transform(x, y, z, self.ax.get_proj())
            ab = AnnotationBbox(imagebox, (x2, y2), xycoords='data', frameon=False)
            self.ax.add_artist(ab)
            return True
        return False

    def _get_node_name(self, uid):
        """[修正] 获取节点的完整名称 (SensorA_0)"""
        if uid is None: return "?"
        node = next((n for n in self.env.nodes if n.node_id == uid), None)
        if node:
            if node.node_type.name == 'TARGET':
                return f"T{node.local_id}"

            # 使用组名 + ID
            base = node.group_name if hasattr(node, 'group_name') and node.group_name else node.node_type.name[0]
            return f"{base}_{node.local_id}"
        return f"#{uid}"

    def render(self, step):
        self.ax.clear()
        self.ax_ui.clear()
        self.ax_ui.axis('off')
        self.ax.set_axis_off()

        # 重绘背景
        self.ax_ui.add_patch(
            patches.Rectangle((0, 0), 1, 1, facecolor=self.sidebar_bg, transform=self.ax_ui.transAxes, zorder=-1))
        self.ax_ui.plot([1, 1], [0, 1], color='#E0E0E0', lw=2)

        # === 3D 场景 ===
        self.ax.set_xlim(0, self.sx)
        self.ax.set_ylim(0, self.sy)
        self.ax.set_zlim(0, self.sz)

        # =========================================================================
        # [核心修正] 底面装饰 (完全移除内部网格，增加战术细节)
        # =========================================================================
        # 网格 (使用 sx, sy 动态绘制)
        for x in np.linspace(0, self.sx, self.gx + 1):
            self.ax.plot([x, x], [0, self.sy], [0, 0], color=self.grid_color, lw=0.8)
        for y in np.linspace(0, self.sy, self.gy + 1):
            self.ax.plot([0, self.sx], [y, y], [0, 0], color=self.grid_color, lw=0.8)

        # Z轴辅助线
        corners = [(0, 0), (self.sx, 0), (0, self.sy), (self.sx, self.sy)]
        for cx, cy in corners:
            self.ax.plot([cx, cx], [cy, cy], [0, self.sz], color=self.grid_color, lw=0.8, ls=':')

        # 1. 绘制底面外框 (仅 4 条边)
        boundary_x = [0, self.sx, self.sx, 0, 0]
        boundary_y = [0, 0, self.sy, self.sy, 0]
        boundary_z = [0, 0, 0, 0, 0]
        self.ax.plot(boundary_x, boundary_y, boundary_z, color=self.grid_color, lw=1.5, alpha=0.8, ls='-')

        # 2. 增加四角“+”字准星装饰 (Corner Accents)
        # 不画网格，只在四个角画标记
        corner_len_x = self.sx * 0.05
        corner_len_y = self.sy * 0.05

        # 左下角
        self.ax.plot([0, corner_len_x], [0, 0], [0, 0], color=self.accent_color, lw=2)
        self.ax.plot([0, 0], [0, corner_len_y], [0, 0], color=self.accent_color, lw=2)
        # 右下角
        self.ax.plot([self.sx, self.sx - corner_len_x], [0, 0], [0, 0], color=self.accent_color, lw=2)
        self.ax.plot([self.sx, self.sx], [0, corner_len_y], [0, 0], color=self.accent_color, lw=2)
        # 右上角
        self.ax.plot([self.sx, self.sx - corner_len_x], [self.sy, self.sy], [0, 0], color=self.accent_color, lw=2)
        self.ax.plot([self.sx, self.sx], [self.sy, self.sy - corner_len_y], [0, 0], color=self.accent_color, lw=2)
        # 左上角
        self.ax.plot([0, corner_len_x], [self.sy, self.sy], [0, 0], color=self.accent_color, lw=2)
        self.ax.plot([0, 0], [self.sy, self.sy - corner_len_y], [0, 0], color=self.accent_color, lw=2)

        # 3. 中心十字准星 (Center Cross)
        cx, cy = self.sx / 2, self.sy / 2
        cross_size = self.sx * 0.03
        self.ax.plot([cx - cross_size, cx + cross_size], [cy, cy], [0, 0], color=self.grid_color, lw=1, alpha=0.5)
        self.ax.plot([cx, cx], [cy - cross_size, cy + cross_size], [0, 0], color=self.grid_color, lw=1, alpha=0.5)

        # 4. 四角立柱 (Corner Pillars) - 辅助空间感，极淡虚线
        corners = [(0, 0), (self.sx, 0), (0, self.sy), (self.sx, self.sy)]
        for c in corners:
            self.ax.plot([c[0], c[0]], [c[1], c[1]], [0, self.sz], color=self.grid_color, lw=0.8, ls=':', alpha=0.3)

        # =========================================================================

        # 智能体
        agents = self.env.nodes[:self.env.n_agents]
        cfg = getattr(self.env, 'cfg', None)
        r_det = getattr(cfg, 'base_detection_range', 5.0) if cfg else 5.0
        r_comm = getattr(cfg, 'base_comm_range', 8.0) if cfg else 8.0

        for i, agent in enumerate(agents):
            if not agent.alive: continue
            x, y, z = agent.pos
            self.trails[i].append((x, y, z))

            ntype = agent.node_type.name
            icon_key = '?'
            if 'SENSOR' in ntype:
                c = self.colors['S'];
                m = '^';
                icon_key = 'S'
            elif 'DECISION' in ntype:
                c = self.colors['D'];
                m = 's';
                icon_key = 'D'
            elif 'INTERCEPTOR' in ntype:
                c = self.colors['I'];
                m = 'd';
                icon_key = 'I'
            else:
                c = 'grey';
                m = 'o'

            if not self._draw_icon_marker(x, y, z, icon_key, c, zoom=0.08):
                self.ax.scatter(x, y, z, c=c, marker=m, s=120, alpha=1.0, edgecolors='white', linewidth=1.5)

            # [修正] 显示完整名称 (SensorA_0)
            label = self._get_node_name(agent.node_id)
            self.ax.text(x, y, z + self.label_offset, label, color=self.text_primary, fontsize=4, fontweight='bold',
                         ha='center')

            # if 'SENSOR' in ntype and r_det > 0:
            #     self._draw_range_circle(x, y, z, r_det, c)

            if len(self.trails[i]) >= 2:
                xs, ys, zs = zip(*self.trails[i])
                self.ax.plot(xs, ys, zs, c=c, alpha=0.4, lw=1.0, ls='-')

        # 通信线
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a1 = agents[i];
                a2 = agents[j]
                if a1.alive and a2.alive:
                    dist = np.linalg.norm(a1.pos - a2.pos)
                    if dist <= r_comm:
                        # 极淡的连线
                        self.ax.plot([a1.pos[0], a2.pos[0]],
                                     [a1.pos[1], a2.pos[1]],
                                     [a1.pos[2], a2.pos[2]],
                                     c='#000000', alpha=0.5, lw=1.5, ls=':')
        # 1. 扫描并捕获闭环
        for t in self.env.tasks:
            # 如果该任务已经彻底展示并销毁过，直接跳过
            if t.task_id in self.finished_chains:
                continue
            if t.stage.name in ['STRIKING', 'COMPLETED']:
                log = getattr(t, 'active_chain_log', None)
                if log:
                    s_n = next((n for n in self.env.nodes if n.node_id == log.get('s')), None)
                    d_n = next((n for n in self.env.nodes if n.node_id == log.get('d')), None)
                    i_n = next((n for n in self.env.nodes if n.node_id == log.get('i')), None)
                    t_n = next((n for n in self.env.nodes if n.node_id == t.target_node_id), None)

                    if all([s_n, d_n, i_n, t_n]):
                        pts = np.array([t_n.pos, s_n.pos, d_n.pos, i_n.pos, t_n.pos])
                        # 记录闭环，并赋予初始寿命。后续循环不会再进这个 block
                        if t.task_id not in self.chain_history or t.stage.name == 'STRIKING':
                            self.chain_history[t.task_id] = {'pts': pts, 'life': 3}
                        if t.stage.name == 'STRIKING':
                            self.ax.plot(
                                [i_n.pos[0], t_n.pos[0]],
                                [i_n.pos[1], t_n.pos[1]],
                                [i_n.pos[2], t_n.pos[2]],
                                color='#FF1744',  # 亮红色
                                lw=2.0,  # 这里的线宽比蓝色(3.0)更粗
                                alpha=0.8,  # 不透明
                                zorder=30,  # 层级最高，压在蓝色线上面
                                ls='-'
                            )

        # 2. 统一绘制与倒计时
        expired_ids = []
        for tid, data in self.chain_history.items():
            pts, life = data['pts'], data['life']
            alpha_val = 0.8 * (life / 3.0)
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='#1E90FF', lw=2.0, alpha=alpha_val, zorder=20)

            # 获取该任务当前实时状态
            curr_t = next((tk for tk in self.env.tasks if tk.task_id == tid), None)

            # 重要：只有当任务【不再是打击状态】（即已 COMPLETED）时，才开始扣减生命值
            if curr_t and curr_t.stage.name != 'STRIKING':
                data['life'] -= 1

            if data['life'] <= 0:
                expired_ids.append(tid)

        # 3. 销毁并将 ID 加入黑名单
        for tid in expired_ids:
            self.chain_history.pop(tid)
            self.finished_chains.add(tid)  # 锁定该任务，本回合不再渲染

        # 目标
        target_map = {t.target_node_id: t for t in self.env.tasks}
        targets = [n for n in self.env.nodes if n.node_type.name == 'TARGET']
        for tgt in targets:
            x, y, z = tgt.pos
            task = target_map.get(tgt.node_id, None)
            status = task.stage.name if task else 'PENDING'
            c = self._get_target_color(status)

            is_completed = (status == 'COMPLETED')

            # [视觉优化] 完成后变透明 (Ghosted)
            alpha_val = 0.3 if is_completed else 1.0

            if not self._draw_icon_marker(x, y, z, 'TARGET', c, zoom=0.06, alpha=alpha_val):
                if is_completed:
                    self.ax.scatter(x, y, z, c='#B0BEC5', marker='x', s=60, lw=2, alpha=0.4)
                else:
                    self.ax.scatter(x, y, z, c=c, marker='D', s=60, edgecolors='white', lw=1.0)

            # Label
            if not is_completed:
                label_text = self._get_node_name(tgt.node_id)  # T0
                if status == 'STRIKING':
                    sub_text = f"Pk:{task.final_kill_prob:.2f}"
                    label_text += f"\n{sub_text}"
                elif status != 'PENDING':
                    sub_text = f"{task.time_window:.0f}s"
                    label_text += f"\n{sub_text}"

                lc = '#90A4AE' if status == 'PENDING' else self.text_primary
                self.ax.text(x, y, z + self.label_offset, label_text, color=lc, fontsize=8, fontweight='bold',
                             ha='center')

        # 交互线 (高亮)
        if hasattr(self.env, 'interaction_history'):
            for act in self.env.interaction_history:
                src = next((n for n in self.env.nodes if n.node_id == act.from_id), None)
                dst = next((n for n in self.env.nodes if n.node_id == act.to_id), None)
                if src and dst:
                    atype = str(act.interaction_type).lower()

                    if 'strike' in atype:
                        print(f"DEBUG: 检测到打击动作! Type={atype}, Src={src.node_id}, Dst={dst.node_id}")
                    # 默认层级
                    z_ord = 10
                    if 'strike' in atype:
                        c_line, l_width, l_style = '#FF1744', 5.0, '-'  # 红色实线
                        z_ord = 30
                    elif 'detect' in atype:  # 使用关键字匹配，更稳妥
                        c_line, l_width, l_style = '#FFD600', 3.0, '-.'  # 黄色点划线
                    elif 'comm' in atype or 'report' in atype:
                        c_line, l_width, l_style = '#00E676', 3.0, '--'  # 绿色虚线
                    else:
                        c_line, l_width, l_style = '#B0BEC5', 4.0, ':'  # 其他未知交互用灰色

                    self.ax.plot([src.pos[0], dst.pos[0]],
                                 [src.pos[1], dst.pos[1]],
                                 [src.pos[2], dst.pos[2]],
                                 c=c_line, lw=l_width, alpha=0.8, ls=l_style,zorder=z_ord)

            # === [关键帧捕获逻辑] ===
            if self.save_frames and hasattr(self.env, 'interaction_history'):
                for act in self.env.interaction_history:
                    # 找到交互关联的任务 ID (假设 act 对象中有相关属性，或者通过目标节点反查)
                    # 这里假设环境传回的任务 ID 为 tid
                    tid = getattr(act, 'task_id', None)
                    if tid is None: continue

                    # 初始化该任务的记录集
                    if tid not in self.triggered_events:
                        self.triggered_events[tid] = set()

                    atype = str(act.interaction_type).lower()

                    # 映射你的动作类型
                    event_map = {
                        'detection': 'detection',
                        'comm': 'communication',
                        'command': 'command_issued',
                        'strike': 'striking'
                    }

                    for key, label in event_map.items():
                        if key in atype and label not in self.triggered_events[tid]:
                            # 先完成当前帧的绘制再截图
                            # (由于 render 是先画图后显示，确保此处在所有 plot 之后执行或使用 canvas.draw)
                            self.fig.canvas.draw()
                            self._capture_frame(tid, label, step)
                            self.triggered_events[tid].add(label)

        # === [UI 侧边栏] ===
        self.ax_ui.text(0.08, 0.94, "SIMULATION DASHBOARD", fontsize=12, fontweight='bold', color=self.text_primary)
        self.ax_ui.text(0.08, 0.91, f"Step: {step}", fontsize=10, color=self.text_secondary)
        self._draw_ui_divider(0.88)

        # Stats
        col1_x = 0.08;
        col2_x = 0.55
        active_cnt = len([t for t in self.env.tasks if t.stage.name not in ['COMPLETED', 'FAILED']])
        self.ax_ui.text(col1_x, 0.84, "THREATS", fontsize=9, color=self.text_secondary)
        self.ax_ui.text(col1_x, 0.81, str(active_cnt), fontsize=16, fontweight='bold', color=self.colors['I'])
        self.ax_ui.text(col2_x, 0.84, "SCORE", fontsize=9, color=self.text_secondary)
        self.ax_ui.text(col2_x, 0.81, f"{self.env.total_reward:.0f}", fontsize=16, fontweight='bold',
                        color=self.accent_color)

        msgs = len(self.env.message_queue)
        completed = self.env.completed_tasks
        self.ax_ui.text(col1_x, 0.76, "MSGS", fontsize=9, color=self.text_secondary)
        self.ax_ui.text(col1_x, 0.73, str(msgs), fontsize=16, fontweight='bold', color=self.text_primary)
        self.ax_ui.text(col2_x, 0.76, "DONE", fontsize=9, color=self.text_secondary)
        self.ax_ui.text(col2_x, 0.73, str(completed), fontsize=16, fontweight='bold', color=self.colors['S'])
        self._draw_ui_divider(0.68)

        # Log
        self.ax_ui.text(0.08, 0.64, "MISSION LOG", fontsize=11, fontweight='bold', color=self.text_primary)
        self.ax_ui.plot([0.08, 0.92], [0.62, 0.62], color=self.divider_color, lw=0.8)

        sorted_tasks = sorted(self.env.tasks, key=lambda t: (t.stage.value, t.task_id))
        y_pos = 0.58
        col_id_x = 0.08;
        col_dot_x = 0.22;
        col_st_x = 0.28;
        col_sub_x = 0.28

        for t in sorted_tasks:
            t_name = self._get_node_name(t.target_node_id)
            st_full = t.stage.name
            tc = self._get_target_color(st_full)

            # 行1
            self.ax_ui.text(col_id_x, y_pos, t_name, fontsize=10, color=self.text_primary, va='center',
                            fontweight='bold')
            self.ax_ui.scatter(col_dot_x, y_pos, color=tc, s=30)
            self.ax_ui.text(col_st_x, y_pos, st_full, fontsize=9, color=tc, va='center', fontweight='bold')

            # 行2
            y_sub = y_pos - 0.035
            info_str = ""
            if st_full == 'COMPLETED':
                log = copy.deepcopy(getattr(t, 'active_chain_log', None))
                if log:
                    s = self._get_node_name(log.get('s'))
                    d = self._get_node_name(log.get('d'))
                    i = self._get_node_name(log.get('i'))
                    # [修正] 闭环格式
                    info_str = f"{t_name}→{s}→{d}→{i}→{t_name}"
                else:
                    info_str = "No Chain Data"
            elif st_full == 'STRIKING':
                info_str = f"Inbound | Pk {t.final_kill_prob:.2f}"
            elif st_full not in ['FAILED', 'PENDING']:
                info_str = f"Time Remaining: {t.time_window:.0f}s"
            elif st_full == 'PENDING':
                info_str = "Scanning..."

            if info_str:
                self.ax_ui.text(col_sub_x, y_sub, info_str, fontsize=7, color=self.text_secondary, va='center',
                                family='monospace')

            y_pos -= 0.075
            if y_pos < 0.05: break

        self.fig.canvas.draw()  # 先渲染出图像

        # [新增] 调用截图检测
        self._save_current_frame(step)
        self.fig.canvas.flush_events()
        plt.pause(self.refresh_interval)

    def _save_current_frame(self, step):
        """[修改] 简单粗暴：保存当前帧"""
        if not self.save_frames:
            return

        # 按照帧号命名，方便后续查阅或合成视频
        # 格式: Frame_0001.svg
        filename = f"Frame_{step:04d}.svg"
        filepath = os.path.join(self.save_dir, filename)

        # 保存 (format='svg' 或 'png'，看你需求)
        self.fig.savefig(filepath, format='svg', bbox_inches='tight')

        # 如果嫌刷屏太快，可以注释掉下面这行
        print(f"\r[Visualizer] Saved: {filename}", end="")

    def close(self):
        plt.ioff()
        plt.show()