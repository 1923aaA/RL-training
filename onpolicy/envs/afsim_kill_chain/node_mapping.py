"""
S/D/I 节点 <-> AFSIM 平台 ID 双向映射器。

这是集成的核心枢纽。
- KillChain 侧用 node_id (0~13) 作为 agent 索引
- AFSIM 侧用 platform_index (整数) 标识平台
- DDS 消息里既有 platform_index,又有 side 名称

映射策略有两种 (通过 AFSIMConfig.role_assignment_mode 切换):

方式 A: 按 side 名称识别 (默认,推荐)
  AFSIM 里约定 side 命名,PlatformAdded 事件来时根据 side 自动分类
  比如:
    side="blue_sensor_0"       -> 分配给 SensorA
    side="blue_interceptor_2"  -> 分配给 LauncherB
  要求 AFSIM 场景里的 side 顺序稳定,每次 reset 分配顺序可重复

方式 B: 显式 platform_index 列表 (适合 AFSIM 场景固定且已知)
  在 AFSIMConfig.explicit_platform_ids 里指定每类节点的平台 ID 列表
  比如 {"SENSOR": [1, 3, 5, 7, 9], ...}
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from onpolicy.envs.afsim_kill_chain.core import NodeType, EndogenousNode


@dataclass
class PlatformInfo:
    """DDS PlatformAdded 消息里能拿到的信息"""
    platform_index: int
    name: str
    side: str
    x_km: float
    y_km: float
    z_km: float


class NodeMapper:
    """
    维护 node_id (0..n_agents+n_target-1) <-> AFSIM platform_index 的双向映射。
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: afsim_kill_chain.config.Config 实例
        """
        self.cfg = cfg
        self.afsim_cfg = cfg.afsim

        # 映射表
        # node_id -> platform_index
        self._node_to_platform: Dict[int, int] = {}
        # platform_index -> node_id
        self._platform_to_node: Dict[int, int] = {}

        # 节点元数据 (node_id -> (node_type, group_name, local_id, capability))
        # 这个表是静态的,按 config 顺序预先生成
        self._node_meta: Dict[int, dict] = {}

        # 已注册的平台信息 (platform_index -> PlatformInfo)
        self._platform_infos: Dict[int, PlatformInfo] = {}

        # 按角色分桶 ("SENSOR"/"DECISION"/"INTERCEPTOR"/"TARGET" -> List[platform_index])
        self._role_to_platforms: Dict[str, List[int]] = {
            "SENSOR": [],
            "DECISION": [],
            "INTERCEPTOR": [],
            "TARGET": [],
        }

        # 按角色的 "下一个 node_id 分配游标"
        # 用于保证分配顺序按 config 里 S_GROUPS 的顺序
        self._role_node_ids: Dict[str, List[int]] = {}

        self._build_static_node_table()

    # -------------------------------------------------------------------
    # 静态节点表构建
    # -------------------------------------------------------------------
    def _build_static_node_table(self):
        """
        按 config 里 S_GROUPS/D_GROUPS/I_GROUPS 的顺序,预分配 node_id。
        node_id 顺序: [所有 S] + [所有 D] + [所有 I] + [所有 TARGET]

        重要: node_id 顺序一旦确定就不变,这是 obs/action 向量里的固定下标。
        """
        node_id_counter = 0

        sensor_ids: List[int] = []
        decision_ids: List[int] = []
        interceptor_ids: List[int] = []

        # S 节点
        for group in self.cfg.S_GROUPS:
            for local_id in range(group["count"]):
                self._node_meta[node_id_counter] = {
                    "node_type": NodeType.SENSOR,
                    "group_name": group["name"],
                    "local_id": local_id,
                    "capability": np.array(group["cap"], dtype=np.float32),
                }
                sensor_ids.append(node_id_counter)
                node_id_counter += 1

        # D 节点
        for group in self.cfg.D_GROUPS:
            for local_id in range(group["count"]):
                self._node_meta[node_id_counter] = {
                    "node_type": NodeType.DECISION,
                    "group_name": group["name"],
                    "local_id": local_id,
                    "capability": np.array(group["cap"], dtype=np.float32),
                }
                decision_ids.append(node_id_counter)
                node_id_counter += 1

        # I 节点
        for group in self.cfg.I_GROUPS:
            for local_id in range(group["count"]):
                self._node_meta[node_id_counter] = {
                    "node_type": NodeType.INTERCEPTOR,
                    "group_name": group["name"],
                    "local_id": local_id,
                    "capability": np.array(group["cap"], dtype=np.float32),
                }
                interceptor_ids.append(node_id_counter)
                node_id_counter += 1

        # Target 节点预留 node_id (具体几个 target 由 env.reset() 时的 curr_n_target 决定)
        target_ids: List[int] = []
        for i in range(self.cfg.max_targets_limit):
            tid = node_id_counter + i
            self._node_meta[tid] = {
                "node_type": NodeType.TARGET,
                "group_name": "Target",
                "local_id": i,
                "capability": np.zeros(4, dtype=np.float32),
            }
            target_ids.append(tid)

        self._role_node_ids = {
            "SENSOR":      sensor_ids,
            "DECISION":    decision_ids,
            "INTERCEPTOR": interceptor_ids,
            "TARGET":      target_ids,
        }

        assert len(sensor_ids) == self.cfg.n_sensor
        assert len(decision_ids) == self.cfg.n_decision
        assert len(interceptor_ids) == self.cfg.n_interceptor
        assert len(sensor_ids) + len(decision_ids) + len(interceptor_ids) == self.cfg.n_agents

    # -------------------------------------------------------------------
    # AFSIM 平台注册 (在 reset 时调用)
    # -------------------------------------------------------------------
    def clear(self):
        """清空所有动态映射, 保留静态 node_meta 表。用在 env.reset() 之前。"""
        self._node_to_platform.clear()
        self._platform_to_node.clear()
        self._platform_infos.clear()
        for role in self._role_to_platforms:
            self._role_to_platforms[role] = []

    def register_platform(self, info: PlatformInfo) -> Optional[int]:
        """
        收到 AFSIM PlatformAdded 时调用, 自动分配一个 node_id。

        Returns:
            分配到的 node_id, 或者 None 如果平台不符合任何角色 / 已满。
        """
        # === 防重入: 该 platform_index 已注册过, 直接返回已有 node_id ===
        if info.platform_index in self._platform_to_node:
            # 顺便更新最新位置信息 (C++ 每秒重发 PlatformAdded 会带最新位姿)
            self._platform_infos[info.platform_index] = info
            return self._platform_to_node[info.platform_index]
        role = self._classify_platform_role(info)
        if role is None:
            # 既不属于蓝方 S/D/I, 也不属于红方 TARGET, 忽略
            return None

        # 看该角色是否还有空位
        available_node_ids = self._role_node_ids[role]
        used_count = len(self._role_to_platforms[role])

        if used_count >= len(available_node_ids):
            # 该角色已经分配满了, 多余平台忽略
            return None

        # 按 "已分配数量" 选下一个 node_id
        node_id = available_node_ids[used_count]

        # 记录映射
        self._node_to_platform[node_id] = info.platform_index
        self._platform_to_node[info.platform_index] = node_id
        self._platform_infos[info.platform_index] = info
        self._role_to_platforms[role].append(info.platform_index)

        return node_id

    def unregister_platform(self, platform_index: int) -> Optional[int]:
        """
        收到 AFSIM PlatformDeleted 时调用。返回该平台对应的 node_id (如果有)。
        注意: 不会回收 node_id, 因为后续可能还会收到该 node 的消息。
        只是标记:以后 update_position 时根据 node.alive 判断。
        """
        if platform_index not in self._platform_to_node:
            return None
        node_id = self._platform_to_node[platform_index]
        # 保留映射表, 调用方可以根据 node.alive=False 禁用
        return node_id

    # -------------------------------------------------------------------
    # 查询接口
    # -------------------------------------------------------------------
    def node_to_platform(self, node_id: int) -> Optional[int]:
        """node_id -> platform_index"""
        return self._node_to_platform.get(node_id)

    def platform_to_node(self, platform_index: int) -> Optional[int]:
        """platform_index -> node_id"""
        return self._platform_to_node.get(platform_index)

    def get_node_meta(self, node_id: int) -> Optional[dict]:
        """返回 node_id 对应的元数据字典"""
        return self._node_meta.get(node_id)

    def get_platform_info(self, platform_index: int) -> Optional[PlatformInfo]:
        return self._platform_infos.get(platform_index)

    def all_registered_node_ids(self) -> List[int]:
        """所有已成功分配了 AFSIM 平台的 node_id (按 node_id 排序)"""
        return sorted(self._node_to_platform.keys())

    def all_registered_target_ids(self) -> List[int]:
        """所有已分配的 TARGET node_id"""
        platform_ids = self._role_to_platforms["TARGET"]
        return [self._platform_to_node[p] for p in platform_ids]

    def is_mapping_ready(self) -> Tuple[bool, str]:
        """
        检查必需的蓝方节点 (S+D+I) 是否全部注册完毕。
        TARGET 不强制要求全部到齐 (可以动态增减)。

        Returns:
            (is_ready, reason_if_not_ready)
        """
        missing = []
        for role in ["SENSOR", "DECISION", "INTERCEPTOR"]:
            expected = len(self._role_node_ids[role])
            actual = len(self._role_to_platforms[role])
            if actual < expected:
                missing.append(f"{role}: {actual}/{expected}")

        if missing:
            return False, "Missing: " + ", ".join(missing)
        return True, ""

    # -------------------------------------------------------------------
    # 构建节点对象 (在 env.reset 时调用)
    # -------------------------------------------------------------------
    def create_endogenous_nodes(self) -> List[EndogenousNode]:
        """
        基于当前已注册的平台, 构建 EndogenousNode 列表。
        顺序: 按 node_id 从 0 开始, 蓝方 S->D->I, 然后 TARGET。

        注意: 如果某个 node_id 没分配到 AFSIM 平台, 会创建一个 alive=False 的占位节点。
        """
        nodes: List[EndogenousNode] = []

        # 蓝方所有 node_id (0 .. n_agents-1)
        for node_id in range(self.cfg.n_agents):
            meta = self._node_meta[node_id]
            platform_index = self._node_to_platform.get(node_id, -1)

            node = EndogenousNode(
                node_id=node_id,
                node_type=meta["node_type"],
                group_name=meta["group_name"],
                local_id=meta["local_id"],
                energy=self.cfg.init_energy,
                alive=(platform_index != -1),
                pos=np.zeros(3, dtype=np.float32),
                raw_capability=meta["capability"].copy(),
                load=0.0,
                theta=1.0,
                afsim_platform_index=platform_index,
            )

            # 如果已注册,初始化 pos
            if platform_index != -1:
                pinfo = self._platform_infos[platform_index]
                node.pos = np.array([pinfo.x_km, pinfo.y_km, pinfo.z_km], dtype=np.float32)
                node.afsim_side = pinfo.side

            nodes.append(node)

        return nodes

    def create_target_nodes(self) -> List[EndogenousNode]:
        """
        基于当前已注册的红方平台,构建 TARGET 节点列表。
        节点数等于 AFSIM 实际报告的红方平台数。
        """
        target_nodes: List[EndogenousNode] = []
        role_platform_ids = self._role_to_platforms["TARGET"]
        for platform_index in role_platform_ids:
            node_id = self._platform_to_node[platform_index]
            meta = self._node_meta[node_id]
            pinfo = self._platform_infos[platform_index]

            node = EndogenousNode(
                node_id=node_id,
                node_type=NodeType.TARGET,
                group_name=meta["group_name"],
                local_id=meta["local_id"],
                energy=0.0,
                alive=True,
                pos=np.array([pinfo.x_km, pinfo.y_km, pinfo.z_km], dtype=np.float32),
                raw_capability=np.zeros(4, dtype=np.float32),
                afsim_platform_index=platform_index,
                afsim_side=pinfo.side,
            )
            target_nodes.append(node)

        return target_nodes

    # -------------------------------------------------------------------
    # 内部工具
    # -------------------------------------------------------------------
    def _classify_platform_role(self, info: PlatformInfo) -> Optional[str]:
        """
        根据 cfg.afsim.role_assignment_mode, 把平台归类到 S/D/I/TARGET。
        返回 None 表示该平台不属于我们关心的任何角色。
        """
        mode = self.afsim_cfg.role_assignment_mode

        if mode == "side":
            # 按 side 前缀匹配
            side_lower = info.side.lower()
            for role, prefix in self.afsim_cfg.side_prefix_map.items():
                if side_lower.startswith(prefix.lower()):
                    return role
            return None

        elif mode == "explicit":
            # 按显式 platform_index 匹配
            for role, pid_list in self.afsim_cfg.explicit_platform_ids.items():
                if info.platform_index in pid_list:
                    return role
            return None

        else:
            raise ValueError(f"Unknown role_assignment_mode: {mode}")

    def summary(self) -> str:
        """Pretty-print 当前映射状态, 用于调试"""
        lines = ["=== NodeMapper Summary ==="]
        for role in ["SENSOR", "DECISION", "INTERCEPTOR", "TARGET"]:
            expected = len(self._role_node_ids[role])
            actual = len(self._role_to_platforms[role])
            lines.append(f"  {role:12s}: {actual}/{expected}")
            for platform_id in self._role_to_platforms[role]:
                node_id = self._platform_to_node[platform_id]
                info = self._platform_infos[platform_id]
                lines.append(f"    node_id={node_id:3d} <-> platform={platform_id:3d} "
                             f"side='{info.side}' name='{info.name}'")
        return "\n".join(lines)