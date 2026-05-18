"""
杀伤链环境配置文件（多尺度场景 + 详细物理注释版）
"""
from dataclasses import dataclass
from typing import Tuple, List, Dict

# ================= 场景选择开关 (Scenario Selector) =================
# 1. 调试场景 (4km, 7智能体): 空间小，密度高，用于快速验证代码无Bug
SCENARIO_DEBUG = "DEBUG"
# 2. 战术场景 (10km, 14智能体): 超过单跳通信距离，强制模型学习中继转发
SCENARIO_MEDIUM = "MEDIUM"
# 3. 战略场景 (20km, 21智能体): 极度稀疏奖励，最终大考 (使用您的完整配置)
SCENARIO_LARGE = "LARGE"

# 【在此处修改当前想要训练的场景】
CURRENT_SCENARIO = SCENARIO_DEBUG
# ===================================================================

@dataclass
class Config:
    """
    杀伤链环境配置类
    包含物理参数、能耗模型、奖励函数及节点编组配置
    """
    # ==================== 0. 神经网络输入维度锚点 (Padding) ====================
    # [关键] 无论当前跑哪个场景，网络都按这个最大规模来初始化输入层维度
    # 作用：让 DEBUG 场景训练出的模型参数维度能和 LARGE 场景对齐，实现无缝加载
    max_agents_limit: int = 30   # 预留给 25~30 个智能体的槽位 (多余补0)
    max_targets_limit: int = 15  # 预留给 15 个目标的槽位

    # ==================== 1. 动态初始化逻辑 ====================
    def __init__(self):
        # --- 通用物理参数 ---
        self.base_detection_range = 1.5   # [探测半径]: 传感器能发现目标的最大距离 (km)
        self.base_comm_range = 2.0        # [通信半径]: 节点间能建立数据链路的最大距离 (km)
        self.base_strike_range = 1.5      # [打击半径]: 拦截器武器能覆盖的杀伤范围 (km)
        self.energy_cost_base = 1.0       # 基础能耗系数 (全局缩放用)

        # ==================== 动作能耗参数 (Energy Costs) ====================
        # 这些参数决定了智能体的"生存压力"，迫使它们优化行动效率

        # 1. 机动能耗
        self.energy_cost_move = 0.1       # [移动能耗]: 模拟无人机/车辆移动一步的燃油或电力消耗
        self.energy_cost_hover = 0.1      # [悬停能耗]: 保持当前位置静止不动的待机功耗 (空中悬停也费电)
        self.energy_cost_collision = 5.0  # [碰撞惩罚]: 发生物理碰撞时的结构性损伤扣血 (高额惩罚以避免自杀式行为)

        # 2. 任务载荷能耗
        self.energy_cost_detection = 0.3  # [探测能耗]: 开启雷达或光电设备进行一次主动扫描的瞬时功耗
        self.energy_cost_comm = 0.5       # [通信能耗]: 发射高功率无线电信号传输情报/指令的能耗 (距离越远越费电)
        self.energy_cost_strike = 0.5     # [打击能耗]: 火控雷达锁定目标及武器发射瞬间的高能耗 (不含导弹自身成本)

        # ==================== 奖励参数 (Rewards) ====================
        # 引导智能体行为的核心指挥棒

        self.shaping_factor = 3.0         # [引导系数]: 距离引导奖励的放大倍数 (让智能体更快趋近目标)
        self.prob_new_task = 0.3          # [任务生成率]: 每步生成新任务的概率
        self.min_kill_probability = 0.6   # [杀伤阈值]: 只有全链路概率 Pk 超过此值，打击才算成功

        # 正反馈
        # self.detection_reward = 1.0       # 发现目标奖励
        self.assignment_reward = 3.0      # 成功指派任务奖励
        self.reward_routing_success = 5.0 # 通信链路建立成功奖励
        self.strike_initiation_reward = 1 # 发起打击动作奖励
        # self.finish_const_reward = 150.0  # [终结大奖]: 成功摧毁目标的巨额奖励
        self.energy_efficiency_reward = 20.0 # 剩余能量奖励 (鼓励省电)

        # 负反馈
        self.w_energy = 0.2               # 能量自然流逝的惩罚权重
        self.w_time = 0.1                 # 时间流逝惩罚 (每拖一步扣一点分，鼓励速战速决)
        self.timeout_penalty = 0.0        # 任务超时未完成的惩罚
        self.failure_penalty = -40.0      # [失败惩罚]: 打击判定失败 (如脱靶) 的重罚
        self.soft_failure_penalty = -2.0  # 软失败惩罚 (如概率不足强行开火)
        self.pending_penalty_per_task = 0.1 # 漏怪惩罚 (场上每存留一个未消灭目标，每步扣分)

        # ==================== 其他模型参数 ====================
        self.base_data_rate = 200.0       # 通信带宽基准 (km/step)
        self.base_decision_time = 4.0     # 指控节点处理决策的基准耗时 (步数)
        self.complexity_factor_per_node = 0.3 # 任务协同复杂度系数
        self.base_weapon_speed = 1.2      # 导弹/拦截弹飞行速度 (km/step)
        self.weapon_speed_per_cap = 0.1   # 能力对武器速度的加成
        self.weapon_cep = 30.0            # 武器圆概率误差 (米)
        self.target_radius = 50.0         # 目标杀伤半径 (米)
        self.jamming_radius = 2.5         # 干扰半径
        self.jamming_intensity = 0.9     # 干扰强度

        # ==================== 场景加载逻辑 ====================
        if CURRENT_SCENARIO == SCENARIO_DEBUG:
            self._setup_debug_scenario()
        elif CURRENT_SCENARIO == SCENARIO_MEDIUM:
            self._setup_medium_scenario()
        elif CURRENT_SCENARIO == SCENARIO_LARGE:
            self._setup_large_scenario()

        # 确保通道限制被正确初始化 (缓冲区大小)
        if not hasattr(self, 'max_channels_S'):
            self.max_channels_S = 3
            self.max_channels_D = 3
            self.max_channels_I = 3

    # ==================== 场景 A: 4x4km (Debug) ====================
    def _setup_debug_scenario(self):
        """最小化配置，仅保留每种类型各 1~2 个，用于跑通代码"""
        self.init_energy = 100.0  # 初始携带能量
        self.task_duration = 40.0 # 任务有效窗口期 (步数)
        self.n_target = 6         # 敌方目标数量

        self.detection_reward = 1.0
        self.finish_const_reward = 150.0  # [终结大奖]: 成功摧毁目标的巨额奖励

        self.space_size_x = 4.0
        self.space_size_y = 4.0
        self.space_size_z = 4.0
        self.grid_size_x = 8
        self.grid_size_y = 8
        self.grid_size_z = 5

        # 总数: 7 个
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

    # ==================== 场景 B: 10x10km (Medium) ====================
    def _setup_medium_scenario(self):
        """中等配置，数量适中，增加任务时长适配地图增大"""
        self.init_energy = 200.0  # 增加能量以支持更远机动
        self.task_duration = 80.0 # 延长时间窗口
        self.n_target = 10

        self.detection_reward = 5.0
        self.finish_const_reward = 400.0  # [终结大奖]: 成功摧毁目标的巨额奖励

        self.space_size_x = 10.0
        self.space_size_y = 10.0
        self.space_size_z = 4.0
        self.grid_size_x = 20
        self.grid_size_y = 20
        self.grid_size_z = 5

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

    # ==================== 场景 C: 20x20km (Large) ====================
    def _setup_large_scenario(self):
        """完全配置，高难度稀疏奖励环境"""
        self.init_energy = 300.0   # 高能量储备
        self.task_duration = 150.0 # 长时间窗口，允许长途奔袭
        self.n_target = 15         # 满载目标

        self.detection_reward = 5.0
        self.finish_const_reward = 400.0  # [终结大奖]: 成功摧毁目标的巨额奖励

        self.space_size_x = 20.0
        self.space_size_y = 20.0
        self.space_size_z = 4.0
        self.grid_size_x = 40
        self.grid_size_y = 40
        self.grid_size_z = 5

        # 总数: 21 个 (完全采用您指定的配置)
        self.S_GROUPS = [
            {"name": "SensorA", "count": 4, "cap": (9.5, 9.0, 2.0, 0.0)}, # 高性能侦察
            {"name": "SensorB", "count": 4, "cap": (6.0, 6.0, 1.0, 0.0)}, # 补盲节点
        ]

        self.D_GROUPS = [
            {"name": "CommandA", "count": 2, "cap": (0.0, 9.5, 9.5, 0.0)},
            {"name": "CommandB", "count": 3, "cap": (0.0, 7.0, 7.0, 0.0)},
        ]

        self.I_GROUPS = [
            {"name": "LauncherA", "count": 3, "cap": (1.0, 6.0, 1.0, 7.5)},
            {"name": "LauncherB", "count": 5, "cap": (2.0, 2.0, 1.0, 9.5)},
        ]

    # ==================== 属性计算 (动态获取总数) ====================
    @property
    def n_agents(self) -> int:
        """根据当前加载的分组动态计算智能体总数"""
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