"""
杀伤链环境核心数据结构（4维能力版本）
定义节点、任务、消息及其交互
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum, IntEnum

class NodeType(Enum):
    """杀伤链中的节点类型"""
    SENSOR = 0       # 侦察节点
    DECISION = 1     # 指挥控制节点
    INTERCEPTOR = 2  # 打击节点
    TARGET = 3       # 对抗目标


class TaskStage(IntEnum):
    """杀伤链任务阶段"""
    PENDING = 0      # 任务生成，未被发现
    DETECTED = 1     # 目标被传感器发现
    ASSIGNED = 2     # 决策处理中/打击任务已分配
    STRIKING = 3     # 武器飞行中
    COMPLETED = 4    # 目标已消灭
    FAILED = 5       #  任务失败或超时

@dataclass
class Message:
    msg_id: str
    initial_source_id: int  # [核心] 原始发起人(S或D)，用于发最终奖励
    from_id: int  # 上一跳是谁(用于计算当前链路距离/概率)
    to_id: int  # 下一跳ID

    # [核心] 目标不再是具体ID，而是类型
    target_node_type: NodeType

    msg_type: str  # "detection_report" / "strike_command"
    content: Dict[str, Any]

    send_time: float
    arrival_time: float

    # [核心] 路径概率累积
    path_prob: float = 1.0  # P_total = P1 * P2 * ...
    hop_count: int = 0  # 跳数
    route_trace: List[int] = field(default_factory=list)  # 防环路记录

    def has_arrived(self, current_time: float) -> bool:
        return current_time >= self.arrival_time

@dataclass
class EndogenousNode:
    """
    内生智能节点（4维能力版本）

    多层结构:
    - Φ (物理层): 位置、能量、存活状态
    - Γ (能力层): [侦察、通信、决策、打击]的基准能力（4维）
    - Δ (活动层): 负载、工作模式
    - Θ (时效层): 随时间衰减的有效性
    """
    node_id: int                # 节点唯一ID
    node_type: NodeType         # 节点类型

    group_name: str  # 所属组名 (如 "SensorA")
    local_id: int   # 组内编号 (如 0, 1, 2)

    # ===== 物理层 (Φ) =====
    energy: float               # 剩余能量
    alive: bool = True          # 存活状态

    # [保留] 物理坐标 (浮点数)，用于计算精确距离
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # 连续速度，单位：km / step。
    # 注意：环境中会根据 speed_*_mps 和 decision_dt 把速度固定到指定模长。
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # ===== 能力层 (Γ) - 4维 =====
    raw_capability: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # ===== 活动层 (Δ) =====
    load: float = 0.0           # 当前负载 [0, 1]

    # ===== 时效层 (Θ) =====
    theta: float = 1.0          # 时效性因子 [0, 1]

    # [新增] 通用数据缓冲区 (实现 FIFO 队列)
    # - S 节点存: 探测到的 target_id (int)
    # - D 节点存: 收到的 task_id (str)
    # - I 节点存: 收到的打击指令消息 cmd_msg (Message)
    data_buffer: List[Any] = field(default_factory=list)

    # [新增] 发送缓冲区：存 S 的情报 / D 的指令 / 别人发来的中继包
    tx_buffer: List[Message] = field(default_factory=list)

    def get_effective_capability(self) -> np.ndarray:
        if not self.alive or self.energy <= 0:
            return np.zeros(4, dtype=np.float32)

        return np.array(self.raw_capability, dtype=np.float32)

    def consume_energy(self, amount: float):
        """
        消耗指定数量的能量

        Args:
            amount: 消耗的能量
        """
        self.energy -= amount
        if self.energy <= 0:
            self.alive = False
            self.energy = 0.0

    def get_actual_ranges(self, cfg):
        """
        根据当前有效能力（受WorkMode调节）计算实际物理半径
        每一个总分为10分，按比例分配。基础距离定义在 config 中。
        """
        # 调用你写好的那个函数
        eff_cap = self.get_effective_capability()

        # 实际距离 = 基础距离 * (当前有效能力 / 10.0)
        actual_det_range = cfg.base_detection_range * (eff_cap[0] / 10.0)
        actual_comm_range = cfg.base_comm_range * (eff_cap[1] / 10.0)
        actual_strike_range = cfg.base_strike_range * (eff_cap[3] / 10.0)

        return actual_det_range, actual_comm_range, actual_strike_range

    def get_load(self, max_capacity: int) -> float:
        """计算负载率 (基于队列长度)"""
        if max_capacity <= 0: return 0.0
        return len(self.data_buffer) / max_capacity

    def reset(self):
        self.alive = True
        self.energy = 100.0
        self.vel = np.zeros(3, dtype=np.float32)
        # pos / vel 会在 Env 的 reset 中重新随机生成。

    def update_position(self, new_pos: np.ndarray):
        """连续坐标更新。pos 单位：km。"""
        self.pos = np.asarray(new_pos, dtype=np.float32)

    def update_velocity(self, new_vel: np.ndarray):
        """连续速度更新。vel 单位：km / step。"""
        self.vel = np.asarray(new_vel, dtype=np.float32)


@dataclass
class Message:
    """
    网络消息（支持多跳盲路由）

    消息在网络中传输需要时间，且携带路由链路信息
    """
    msg_id: str  # 消息唯一ID

    # [新增] 路由与溯源字段
    initial_source_id: int  # 原始发起者ID (用于发奖励)

    from_id: int  # 上一跳发送节点ID
    to_id: int  # 下一跳接收节点ID (路由过程中的临时目标)

    # [新增] 盲路由目标类型
    target_node_type: NodeType  # 最终想找的节点类型 (如 S找DECISION)

    msg_type: str  # 消息类型: "detection_report", "strike_command"
    content: Dict[str, Any]  # 消息内容

    send_time: float  # 发送时间
    arrival_time: float  # 到达时间

    # [新增] 链路追踪字段
    path_prob: float = 1.0  # 链路累积概率 (P_total = P1 * P2 * ...)
    hop_count: int = 0  # 跳数计数
    route_trace: List[int] = field(default_factory=list)  # 记录经过的节点ID

    def has_arrived(self, current_time: float) -> bool:
        """检查消息是否已到达"""
        return current_time >= self.arrival_time


@dataclass
class TaskEntity:
    """
    任务实体（带通信时延版本）
    """
    task_id: str                # 任务唯一ID
    target_node_id: int         # 目标节点ID

    # 时间相关
    time_window: float          # 剩余时间窗口
    total_time: float           # 总时间窗口
    start_time: float           # 开始时间

    first_detection_time: Optional[float] = None

    # 任务阶段
    stage: TaskStage = TaskStage.PENDING

    # 阶段计时
    stage_entry_time: float = 0.0

    # 链路记录
    assigned_s_list: List[int] = field(default_factory=list)  # 探测过的 S
    assigned_d: int = -1  # 负责指控的 D
    assigned_i_list: List[int] = field(default_factory=list)  # 负责打击的 I (单链模式下长度最大为1)

    # [AFSIM 新增] 记住哪个 I 节点实际发射了武器, 等 STRIKE_RESULT (topic 5002) 回来时分奖励
    # 纯公式版不设, AFSIM 版在 _perform_strike_logic 里赋值
    # 向后兼容: 旧代码不碰这个字段, 保持 None 不影响任何行为
    shooter_node_id: Optional[int] = None

    # 结果数据
    final_kill_prob: float = 0.0
    completion_time: float = 0.0
    accumulated_value: float = 0.0

    # 战报日志 (用于记录生效的那条单链的详细参数)
    active_chain_log: Dict[str, Any] = field(default_factory=dict)

    # 奖励暂存
    pending_reward_distribution: Optional[Tuple] = None

    def update_time(self, dt: float = 1.0) -> bool:
        self.time_window -= dt
        return self.time_window <= 0

    def add_sensor_detection(self, sensor_id: int):
        if sensor_id not in self.assigned_s_list:
            self.assigned_s_list.append(sensor_id)

    def set_stage(self, new_stage: TaskStage, current_step: float):
        self.stage = new_stage
        self.stage_entry_time = current_step


@dataclass
class Interaction:
    """
    节点间交互记录
    """
    from_id: int           # 发起节点ID
    to_id: int             # 目标节点ID
    from_type: NodeType    # 发起节点类型
    to_type: NodeType      # 目标节点类型
    distance: float        # 节点间距离
    interaction_type: str  # 交互类型: "detect", "report", "command", "strike"

    # [新增] 负载数据，用于传递 target_id (S->D) 或 task_id (D->I)
    payload: Optional[Any] = None
