"""
杀伤链环境配置文件（连续三维运动版，无网格字段）

说明：
1. 当前默认训练 4 km × 4 km × 4 km 的 DEBUG 场景。
2. 保留原来的随机目标数量逻辑：DEBUG 场景 n_target=6，因此训练时默认随机 4~6 个目标。
3. 不再保留 grid_size_x / grid_size_y / grid_size_z 等网格参数。
4. 本版只围绕奖励/惩罚权重做调整，目的是提高完整杀伤链闭环完成率，并缓解训练后期性能回落。
"""
from dataclasses import dataclass

I = 6

# ================= 场景选择开关 (Scenario Selector) =================
SCENARIO_DEBUG = "DEBUG"    # 4km, 7智能体
SCENARIO_MEDIUM = "MEDIUM"  # 10km, 14智能体
SCENARIO_LARGE = "LARGE"    # 20km, 21智能体

# 当前训练场景：4km × 4km × 4km 小场景
CURRENT_SCENARIO =SCENARIO_DEBUG
# ===================================================================


@dataclass
class Config:
    """
    杀伤链环境配置类。
    环境坐标单位为 km；速度配置单位为 m/s。
    """

    # ==================== 0. 神经网络输入维度锚点 Padding ====================
    max_agents_limit: int = 30
    max_targets_limit: int = 15

    # ==================== 1. 动态初始化逻辑 ====================
    def __init__(self):
        # ==================== 基础物理范围参数 ====================
        # 本次不改动基础探测/通信/打击范围，保持原始任务难度。
        self.base_detection_range = 1.5
        self.base_comm_range = 2.0
        self.base_strike_range = 1.5
        self.energy_cost_base = 1.0

        # ==================== 动作能耗参数 ====================
        self.energy_cost_move = 0.1
        self.energy_cost_hover = 0.1
        self.energy_cost_collision = 5.0
        self.energy_cost_detection = 0.3
        self.energy_cost_comm = 0.5
        self.energy_cost_strike = 0.5

        # ==================== 连续飞行运动学参数 ====================
        self.decision_dt = 2.0

        # 三类己方节点当前统一固定飞行速度为 50 m/s。
        self.speed_S_mps = 50.0
        self.speed_D_mps = 50.0
        self.speed_I_mps = 50.0

        # 最大转弯半径，单位 m。
        self.max_turn_radius_m = 1000.0

        # 连续空间最小安全间隔，单位 km。0.05 km = 50 m。
        self.min_separation_km = 0.05

        # 连续动作高斯头输出裁剪范围。
        self.velocity_action_clip = 1.0

        # ==================== 奖励参数：提高闭环完成率 ====================
        # shaping_factor：增强接近目标/任务相关 shaping，帮助连续速度策略更快找到有效方向。
        self.shaping_factor = 5.0

        # 保留原始任务生成概率。
        self.prob_new_task = 0.3

        # 保留原始最小杀伤概率门限，不通过降低门限人为抬高成功率。
        self.min_kill_probability = 0.5

        # D 节点完成任务分配的奖励。提高该值，鼓励情报到 D 后尽快形成打击任务。
        self.assignment_reward = 5.0

        # 消息成功送达目标类型节点的路由奖励。提高该值，强化 S->D 与 D->I 的通信闭环。
        self.reward_routing_success = 5.0

        # I 节点发起打击的奖励。提高该值，避免 I 节点过度保守。
        self.strike_initiation_reward = 10.0

        # 能效奖励下调。当前主要目标是完成率，不让节能目标过早压制机动和通信。
        self.energy_efficiency_reward = 2.0

        # 能量/时间惩罚权重下调，缓解后期策略为了少扣分而保守的问题。
        self.w_energy = 0.02
        self.w_time = 0.02

        # 超时惩罚设为负值，避免模型接受“不完成任务也没事”的局部最优。
        self.timeout_penalty = -8.0

        # 失败惩罚降低绝对值，避免 I 节点因为担心失败而不敢打。
        self.failure_penalty = -15.0
        self.soft_failure_penalty = -0.5

        # 未完成任务惩罚略降低，避免随机目标数较多时早期训练被负反馈淹没。
        self.pending_penalty_per_task = 0.12

        # ==================== 其他模型参数 ====================
        self.base_data_rate = 200.0
        self.base_decision_time = 4.0
        self.complexity_factor_per_node = 0.3
        self.base_weapon_speed = 1.2
        self.weapon_speed_per_cap = 0.1
        self.weapon_cep = 30.0
        self.target_radius = 50.0
        self.jamming_radius = 2.5
        self.jamming_intensity = 0.0

        # ==================== 场景加载逻辑 ====================
        if CURRENT_SCENARIO == SCENARIO_DEBUG:
            self._setup_debug_scenario()
        elif CURRENT_SCENARIO == SCENARIO_MEDIUM:
            self._setup_medium_scenario()
        elif CURRENT_SCENARIO == SCENARIO_LARGE:
            self._setup_large_scenario()
        else:
            raise ValueError(f"Unknown scenario: {CURRENT_SCENARIO}")

        # ==================== 通道限制 ====================
        if not hasattr(self, "max_channels_S"):
            self.max_channels_S = 3
            self.max_channels_D = 3
            self.max_channels_I = 3

        # ==================== 随机目标数量训练/评估 ====================
        # 保持原来的目标随机逻辑：
        # DEBUG: n_target=6  -> 随机 4~6
        # MEDIUM/LARGE: n_target=15 -> 随机 4~15
        self.use_random_n_target = True
        self.train_n_target_min = max(1, min(4, self.n_target))
        self.train_n_target_max = min(self.n_target, self.max_targets_limit)
        self.eval_n_target_list = sorted(list(set([
            v for v in [4, 6, 8, 10, 12, self.n_target]
            if 1 <= v <= self.max_targets_limit
        ])))

        # ==================== 观测统计特征 ====================
        self.stats_feat_dim = 6

        # ==================== 奖励归一化与进度奖励 ====================
        self.use_ratio_pending_penalty = True
        self.pending_penalty_scale = self.pending_penalty_per_task

        # 提高进度奖励，让模型更偏向“多完成目标”，而不是只完成少数高 Pk 目标。
        self.use_progress_reward = True
        self.progress_reward_scale = 100.0

    # ==================== 场景 A: 4 km × 4 km × 4 km ====================
    def _setup_debug_scenario(self):
        """小场景配置：7 个智能体，最多 6 个目标，训练时随机 4~6 个目标。"""
        self.init_energy = 150.0
        self.task_duration = 120.0
        self.n_target = 6

        # 提高探测奖励，强化杀伤链第一步。
        self.detection_reward = 5.0

        # 提高完成奖励，直接拉高完整闭环任务的优势。
        self.finish_const_reward = 500.0

        self.space_size_x = 4.0
        self.space_size_y = 4.0
        self.space_size_z = 4.0

        self.S_GROUPS = [
            {"name": "SensorA", "count": 1, "cap": (9.5, 9.0, 2.0, 0.0)},
            {"name": "SensorB", "count": 1, "cap": (6.0, 6.0, 1.0, 0.0)},
        ]

        self.D_GROUPS = [
            {"name": "CommandA", "count": 1, "cap": (0.0, 9.5, 9.5, 0.0)},
            {"name": "CommandB", "count": 1, "cap": (0.0, 7.0, 7.0, 0.0)},
        ]

        self.I_GROUPS = [
            {"name": "LauncherA", "count": 1, "cap": (1.0, 6.0, 1.0, 7.5)},
            {"name": "LauncherB", "count": 2, "cap": (2.0, 2.0, 1.0, 9.5)},
        ]

    # ==================== 场景 B: 10 km × 10 km × 4 km ====================
    def _setup_medium_scenario(self):
        """中等场景配置。"""
        self.init_energy = 200.0
        self.task_duration = 200.0
        self.n_target = 15

        self.detection_reward = 6.0
        self.finish_const_reward = 500.0

        self.space_size_x = 10.0
        self.space_size_y = 10.0
        self.space_size_z = 4.0

        self.S_GROUPS = [
            {"name": "SensorA", "count": 3, "cap": (9.5, 9.0, 2.0, 0.0)},
            {"name": "SensorB", "count": 2, "cap": (6.0, 6.0, 1.0, 0.0)},
        ]

        self.D_GROUPS = [
            {"name": "CommandA", "count": 2, "cap": (0.0, 9.5, 9.5, 0.0)},
            {"name": "CommandB", "count": 2, "cap": (0.0, 7.0, 7.0, 0.0)},
        ]

        self.I_GROUPS = [
            {"name": "LauncherA", "count": 2, "cap": (1.0, 6.0, 1.0, 7.5)},
            {"name": "LauncherB", "count": 3, "cap": (2.0, 2.0, 1.0, 9.5)},
        ]

    # ==================== 场景 C: 20 km × 20 km × 4 km ====================
    def _setup_large_scenario(self):
        """大场景配置。"""
        self.init_energy = 300.0
        self.task_duration = 150.0
        self.n_target = 15

        self.detection_reward = 6.0
        self.finish_const_reward = 900.0

        self.space_size_x = 20.0
        self.space_size_y = 20.0
        self.space_size_z = 4.0

        self.S_GROUPS = [
            {"name": "SensorA", "count": 4, "cap": (9.5, 9.0, 2.0, 0.0)},
            {"name": "SensorB", "count": 4, "cap": (6.0, 6.0, 1.0, 0.0)},
        ]

        self.D_GROUPS = [
            {"name": "CommandA", "count": 2, "cap": (0.0, 9.5, 9.5, 0.0)},
            {"name": "CommandB", "count": 3, "cap": (0.0, 7.0, 7.0, 0.0)},
        ]

        self.I_GROUPS = [
            {"name": "LauncherA", "count": 3, "cap": (1.0, 6.0, 1.0, 7.5)},
            {"name": "LauncherB", "count": 5, "cap": (2.0, 2.0, 1.0, 9.5)},
        ]

    # ==================== 属性计算 ====================
    @property
    def n_agents(self) -> int:
        """根据当前加载的分组动态计算智能体总数。"""
        return sum(g["count"] for g in (self.S_GROUPS + self.D_GROUPS + self.I_GROUPS))

    @property
    def n_sensor(self) -> int:
        return sum(g["count"] for g in self.S_GROUPS)

    @property
    def n_decision(self) -> int:
        return sum(g["count"] for g in self.D_GROUPS)

    @property
    def n_interceptor(self) -> int:
        return sum(g["count"] for g in self.I_GROUPS)

    @property
    def n_total_nodes(self) -> int:
        return self.n_agents + self.n_target
