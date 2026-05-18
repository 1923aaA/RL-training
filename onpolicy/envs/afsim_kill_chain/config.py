"""
AFSIM 杀伤链环境配置
从原 kill_chain/config.py 重构而来。

主要变化:
1. 删除了 space_size/grid_size (原 KillChain 用的是 Python 内部网格空间)
2. 增加了 AFSIM + DDS 相关配置 (domain_id, sleep_seconds, 平台角色分配等)
3. 保留了 S/D/I 分组、capability、奖励参数、概率参数 (业务逻辑)
4. Python 决策侧使用连续三维速度动作；AFSIM 侧仍通过原 EntityMoveTo 位置命令交互
5. [AFSIM 集成] episode 间使用 SimRestart；Python 固定 zone1 中心作为局部 ENU 原点
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Dict

import numpy as np


# ================= 场景选择开关 =================
SCENARIO_DEBUG = "DEBUG"    # 4km, 7 agents
SCENARIO_MEDIUM = "MEDIUM"  # 10km, 14 agents (默认)
SCENARIO_LARGE = "LARGE"    # 20km, 21 agents

CURRENT_SCENARIO = SCENARIO_MEDIUM
# =================================================


@dataclass
class AFSIMConfig:
    """
    AFSIM + DDS 相关配置。
    这部分是原 KillChain 没有的全新配置。
    """
    # DDS Domain
    domain_id: int = 40

    # 时序: AFSIM real-time + MAPPO 每 step 等 1 秒
    # [对齐 v3] step_wait_sec=1.0 配合 C++ 端的 WsfQuadrotorMover::TeleportTo:
    # 每个 EntityMoveTo 都让四旋翼瞬移到目标位置 + 悬停, 不再依赖巡航速度飞行.
    # 因此 step_wait_sec 不需要再拖到 2.5s 来补偿物理飞行慢的问题.
    # 1.0s 给 AFSIM 留时间处理 5001/5002 等周期事件即可.
    step_wait_sec: float = 1.0        # env.step() 里 sleep 的时间
    reset_wait_sec: float = 15.0      # env.reset() 后等 AFSIM 平台就绪的时间
    reset_timeout_sec: float = 60.0   # reset 最长等待 (超过就报错)

    # 期望的蓝方平台数量 (必须与 Config.n_agents 一致,用于 sanity check)
    expected_blue_platforms: int = 14

    # 平台角色分配策略
    # 方式 A (默认,推荐): 按 AFSIM 的 side 字段识别
    #   要求 AFSIM 场景里的 side 命名满足:
    #     "blue_sensor_*"       -> SENSOR
    #     "blue_command_*"      -> DECISION
    #     "blue_interceptor_*"  -> INTERCEPTOR
    #     "red_*"               -> TARGET
    # 方式 B: 显式指定 platform_index 列表 (如果你 AFSIM 场景的 side 命名不方便改)
    role_assignment_mode: str = "explicit"   # "side" 或 "explicit"

    # 仅 role_assignment_mode == "side" 时使用
    side_prefix_map: Dict[str, str] = field(default_factory=lambda: {
        "SENSOR":      "blue_sensor",
        "DECISION":    "blue_command",
        "INTERCEPTOR": "blue_interceptor",
        "TARGET":      "red",
    })

    # 仅 role_assignment_mode == "explicit" 时使用
    # 列表内的值是 AFSIM platform_index (整数)  24个平台 (14蓝 + 10红)
    explicit_platform_ids: Dict[str, List[int]] = field(default_factory=lambda: {
        "SENSOR":      [1, 2, 3, 4, 5],
        "DECISION":    [6, 7, 8, 9],
        "INTERCEPTOR": [10, 11, 12, 13, 14],
        "TARGET":      [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    })

    # DDS 桥接参数
    dds_queue_size: int = 10000
    dds_debug_log: bool = False

    # ==================== Python 侧消息交互详细日志 ====================
    # interaction_debug_log=True 时，打印 AFSIM/RL 消息闭环关键节点：
    # 5001 TrackEvent、S 生成 detection_report、SendTrackMessage 下发、pending 登记、
    # 5003 到达、D/I buffer 入库、D 生成 strike_command、FireWeapon 下发等。
    # motion_debug_log 单独控制 EntityMoveTo 批量移动日志，默认关闭，避免训练时刷屏。
    interaction_debug_log: bool = True
    motion_debug_log: bool = False
    log_pending_table_size: bool = True
    track_event_log_every_n: int = 1
    move_log_every_steps: int = 10


    # C++ 侧 StateObserver.cpp 是否已应用 "改动A" (PlatformAdded/MoveUpdate 消息里加了 name/side)
    # 默认 True (假设已改). 若没改,必须设为 False, 同时 role_assignment_mode 必须切 "explicit"
    platform_msg_has_name_side: bool = True

    # ==================== [AFSIM 新增] SimReset 随机化参数 ====================
    # 位置盒子 (LLA, 经纬度单位: 度, 高度单位: 米)
    # 默认值是 PROJECT_STATUS 里 Taiwan 场景的 10x10 km 区域
    # 23:06:03.30N ~ 23:11:26.70N, 120:00:34.15E ~ 120:06:25.85E
    blue_lat_min: float = 23.10092
    blue_lat_max: float = 23.19075
    blue_lon_min: float = 120.00949
    blue_lon_max: float = 120.10718
    blue_alt_min: float = 500.0
    blue_alt_max: float = 3500.0

    red_lat_min: float = 23.10092
    red_lat_max: float = 23.19075
    red_lon_min: float = 120.00949
    red_lon_max: float = 120.10718
    red_alt_min: float = 300.0
    red_alt_max: float = 3200.0

    # soft_reset 发完 SimReset 之后等 AFSIM 处理并重发 PlatformAdded 的时间
    reset_post_sleep_sec: float = 2.0

    # track cache 过期秒数 (local time, 超过这个就从 cache 剔除)
    track_stale_sec: float = 5.0

    # 当前版本不再用 SimReset 在线随机撒点。episode 之间由 SimRestart 触发 AFSIM 原生重新运行。
    reset_randomize_positions: bool = False

    # 发 SimRestart 后给 AFSIM 完成 RequestReset / 场景重新运行的等待时间。
    restart_post_sleep_sec: float = 3.0

    # Python 侧固定局部 ENU 坐标原点：zone1 中心，高度原点为 0 m。
    # 23:06:03.30N ~ 23:11:26.70N, 120:00:34.15E ~ 120:06:25.85E
    use_fixed_lla_origin: bool = True
    fixed_origin_lat: float = 23.145833333333333
    fixed_origin_lon: float = 120.05833333333334
    fixed_origin_alt: float = 0.0


@dataclass
class Config:
    """
    KillChain 业务参数 + AFSIM 集成参数。

    业务逻辑部分 (S/D/I 分组、奖励、概率) 基本照搬原 KillChain。
    删掉网格决策逻辑；空间边界只用于归一化和连续速度积分的边界保护。
    """

    # ==================== 神经网络输入维度锚点 ====================
    # 无论当前跑哪个场景, 网络按这个最大规模初始化输入层
    max_agents_limit: int = 30
    max_targets_limit: int = 15

    def __init__(self):
        # ==================== 通信 / 探测 / 打击半径 (单位:km) ====================
        # 这些值会被 get_actual_ranges(cfg) 用来算每个 agent 的实际半径
        # 原 KillChain 是 base * (cap/10), 继续沿用
        self.base_detection_range = 1.5   # km
        self.base_comm_range = 2.0        # km
        # [对齐] base_strike_range 回退到 1.5 km, 与 SAMAPPO_Ex10_10_4_PLUS 训练 config 一致.
        # (方式 B 训练时曾改成 5.0 km 缓解稀疏奖励, 但加载训练好的 KillChain 模型推理时
        #  必须保持训练分布: actual_strike_range = 1.5 * cap[3]/10 ≈ 1.1-1.4 km)
        self.base_strike_range = 1.5      # km
        self.energy_cost_base = 1.0

        # ==================== 动作能耗参数 ====================
        self.energy_cost_move = 0.1
        self.energy_cost_hover = 0.1
        self.energy_cost_collision = 5.0

        # ==================== 连续飞行运动学参数 ====================
        # 环境坐标单位为 km；速度配置单位为 m/s。
        self.decision_dt = 2.0
        self.speed_S_mps = 50.0
        self.speed_D_mps = 50.0
        self.speed_I_mps = 50.0
        self.max_turn_radius_m = 1000.0
        self.min_separation_km = 0.05
        self.velocity_action_clip = 1.0

        # 固定训练区域边界：zone1 = 10 km × 10 km × 4 km。
        # Bridge 固定 LLA 原点为 zone1 中心后，局部坐标应满足：
        #   x/y ∈ [-5, 5] km，z ∈ [0, 4] km。
        # 这些边界只用于 Python 侧动作积分限幅；AFSIM 侧仍按脚本和自身 mover 运行。
        self.afsim_motion_min_km = (-5.0, -5.0, 0.0)
        self.afsim_motion_max_km = (5.0, 5.0, 4.0)
        self.use_centered_afsim_bounds = False
        self.afsim_bound_slack = 1.0

        self.energy_cost_detection = 0.3
        self.energy_cost_comm = 0.5
        self.energy_cost_strike = 0.5

        # ==================== 奖励参数 ====================
        self.shaping_factor = 3.0
        self.prob_new_task = 0.3
        self.min_kill_probability = 0.6

        self.assignment_reward = 3.0
        self.reward_routing_success = 5.0
        self.strike_initiation_reward = 4
        self.energy_efficiency_reward = 10.0

        self.w_energy = 0.2
        self.w_time = 0.1
        self.timeout_penalty = -20.0
        self.failure_penalty = -20.0
        self.soft_failure_penalty = -2.0
        self.pending_penalty_per_task = 0.1

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

        # ==================== [AFSIM 新增] 打击事件奖励 ====================
        # 对应 PROJECT_STATUS 里的 r_strike_hit / r_strike_miss
        # WEAPON_HIT + targetKilled=1 → reward_strike_hit (给 shooter 独享一份)
        # WEAPON_MISS / WEAPON_TERMINATED 兜底 → reward_strike_miss (给 shooter)
        self.reward_strike_hit = 10.0
        self.reward_strike_miss = -2.0

        # ==================== AFSIM 集成参数 ====================
        self.afsim = AFSIMConfig()

        # ==================== 场景加载 ====================
        if CURRENT_SCENARIO == SCENARIO_DEBUG:
            self._setup_debug_scenario()
        elif CURRENT_SCENARIO == SCENARIO_MEDIUM:
            self._setup_medium_scenario()
        elif CURRENT_SCENARIO == SCENARIO_LARGE:
            self._setup_large_scenario()

        # 通道限制
        if not hasattr(self, 'max_channels_S'):
            self.max_channels_S = 3
            self.max_channels_D = 3
            self.max_channels_I = 3

        # ==================== 可变目标数量 ====================
        self.use_random_n_target = True
        self.train_n_target_min = max(1, min(4, self.n_target))
        self.train_n_target_max = min(self.n_target, self.max_targets_limit)
        self.eval_n_target_list = sorted(list(set([
            v for v in [4, 6, 8, 10, 12, self.n_target]
            if 1 <= v <= self.max_targets_limit
        ])))

        # 观测统计特征
        self.stats_feat_dim = 6

        # 奖励归一化
        self.use_ratio_pending_penalty = True
        self.pending_penalty_scale = self.pending_penalty_per_task
        self.use_progress_reward = True
        self.progress_reward_scale = 20.0

        # 同步期望蓝方数量到 AFSIM 配置
        self.afsim.expected_blue_platforms = self.n_agents

    # ==================== 场景 A: 4km (Debug, 7 agents) ====================
    def _setup_debug_scenario(self):
        self.init_energy = 100.0
        self.task_duration = 40.0
        self.n_target = 6

        self.detection_reward = 1.0
        self.finish_const_reward = 150.0

        # AFSIM 坐标归一化用 (原 KillChain 的 space_size)
        # 只用于 obs normalization, 不用于物理仿真
        self.world_bounds_km = (4.0, 4.0, 4.0)
        # AFSIM 局部 ENU 归一化边界；不再代表网格步长。

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

    # ==================== 场景 B: 10km (Medium, 14 agents) ====================
    def _setup_medium_scenario(self):
        self.init_energy = 500.0
        self.task_duration = 200.0
        # [修正] 原来是 15, 但 explicit_platform_ids["TARGET"] 只列了 10 个 (15-24)
        # 以及 PROJECT_STATUS 写明是 "10 red target". 对齐为 10.
        self.n_target = 10

        self.detection_reward = 10.0
        self.finish_const_reward = 1400.0

        self.world_bounds_km = (10.0, 10.0, 4.0)

        # 总数: 14 个
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

    # ==================== 场景 C: 20km (Large, 21 agents) ====================
    def _setup_large_scenario(self):
        self.init_energy = 300.0
        self.task_duration = 150.0
        self.n_target = 15

        self.detection_reward = 5.0
        self.finish_const_reward = 800.0

        self.world_bounds_km = (20.0, 20.0, 4.0)

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
        return sum(g['count'] for g in (self.S_GROUPS + self.D_GROUPS + self.I_GROUPS))

    @property
    def n_sensor(self) -> int:
        return sum(g['count'] for g in self.S_GROUPS)

    @property
    def n_decision(self) -> int:
        return sum(g['count'] for g in self.D_GROUPS)

    @property
    def n_interceptor(self) -> int:
        return sum(g['count'] for g in self.I_GROUPS)

    @property
    def n_total_nodes(self) -> int:
        return self.n_agents + self.n_target

    # ==================== 兼容旧 API ====================
    # 原 KillChain 代码里用 space_size_x/y/z,这里给兼容访问
    @property
    def space_size_x(self) -> float:
        return self.world_bounds_km[0]

    @property
    def space_size_y(self) -> float:
        return self.world_bounds_km[1]

    @property
    def space_size_z(self) -> float:
        return self.world_bounds_km[2]
