"""
AFSIM 杀伤链环境核心数据结构。
从原 kill_chain/core.py 改造:
- NodeType / TaskStage / Message / TaskEntity / Interaction: 直接重用父类定义
  (保证继承 KillChainEnv 时 enum 对象一致, dict 查找不出错)
- EndogenousNode: 本地重新定义, 添加 AFSIM 关联字段和新的位置更新方法

[AFSIM 新增]
- EndogenousNode.last_known_track_id: I 节点记住最近一次发射对应的 track_id_str,
  用于 STRIKE_RESULT 反馈到来时定位是哪条打击
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any

# [重要] 直接从父类 kill_chain.core 重导出, 保证是同一个 Python 对象
from onpolicy.envs.kill_chain.core import (
    NodeType,
    TaskStage,
    Message,
    TaskEntity,
    Interaction,
)


@dataclass
class EndogenousNode:
    """
    内生智能节点 (AFSIM 适配版)

    和父类 kill_chain.core.EndogenousNode 的区别:
    - 添加 vel 字段 (单位 km/step)，用于连续三维速度动作和观测
    - 添加 afsim_platform_index (关联 AFSIM 里的平台 ID)
    - 添加 afsim_side (AFSIM 里的 side 标识, 用于调试)
    - 添加 afsim_last_update_time (记录最近一次 DDS 更新时间)
    - 新增 update_position_from_afsim 方法，同步 AFSIM 回传的连续坐标
    - [AFSIM 新增] last_known_track_id: I 节点记录本次/最近一次打击的 track_id_str
    """
    node_id: int
    node_type: NodeType

    group_name: str
    local_id: int

    # ===== 物理层 (Φ) =====
    energy: float
    alive: bool = True
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # 连续速度，单位 km/step。vx/vy/vz 会进入 obs/share_obs。
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # 兼容旧调试代码的保留字段；连续动作版不再依赖它做决策。
    grid_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=int))

    # ===== 能力层 (Γ) - 4 维 =====
    raw_capability: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # ===== 活动层 (Δ) =====
    load: float = 0.0

    # ===== 时效层 (Θ) =====
    theta: float = 1.0

    # ===== Buffers =====
    data_buffer: List[Any] = field(default_factory=list)
    tx_buffer: List[Message] = field(default_factory=list)

    # ===== AFSIM 关联 =====
    afsim_platform_index: int = -1
    afsim_side: str = ""
    afsim_last_update_time: float = 0.0

    # [AFSIM 新增] I 节点发射武器后记住触发的 track_id, 便于调试/日志;
    # 正式的 engagement 映射由 env.active_engagements 管理, 这里只是冗余记录
    last_known_track_id: str = ""

    def get_effective_capability(self) -> np.ndarray:
        if not self.alive or self.energy <= 0:
            return np.zeros(4, dtype=np.float32)
        return np.array(self.raw_capability, dtype=np.float32)

    def consume_energy(self, amount: float):
        self.energy -= amount
        if self.energy <= 0:
            self.alive = False
            self.energy = 0.0

    def get_actual_ranges(self, cfg):
        """
        基于当前有效能力计算实际物理半径 (km)。
        完全沿用原 KillChain 公式: actual = base * (cap / 10)
        """
        eff_cap = self.get_effective_capability()
        actual_det_range = cfg.base_detection_range * (eff_cap[0] / 10.0)
        actual_comm_range = cfg.base_comm_range * (eff_cap[1] / 10.0)
        actual_strike_range = cfg.base_strike_range * (eff_cap[3] / 10.0)
        return actual_det_range, actual_comm_range, actual_strike_range

    def get_load(self, max_capacity: int) -> float:
        if max_capacity <= 0:
            return 0.0
        return len(self.data_buffer) / max_capacity

    def reset(self):
        self.alive = True
        self.energy = 100.0
        self.vel = np.zeros(3, dtype=np.float32)
        # pos 不在这里重置, 由 AFSIM 的 PlatformAdded/MoveUpdate 提供

    def update_position(self, new_pos: np.ndarray):
        """连续坐标更新。pos 单位：km。"""
        self.pos = np.asarray(new_pos, dtype=np.float32)

    def update_velocity(self, new_vel: np.ndarray):
        """连续速度更新。vel 单位：km/step。"""
        self.vel = np.asarray(new_vel, dtype=np.float32)

    def update_position_from_afsim(self, pos_xyz_km: np.ndarray,
                                   timestamp: float = 0.0):
        """
        从 AFSIM DDS MoveUpdate 消息更新连续位置。

        Args:
            pos_xyz_km: AFSIM 给的位置, 单位 km (x, y, z)
            timestamp: AFSIM 仿真时间戳, 可选
        """
        self.pos = np.array(pos_xyz_km, dtype=np.float32)
        self.afsim_last_update_time = timestamp
