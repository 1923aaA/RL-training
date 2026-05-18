"""
AFSIM Kill Chain Environment (连续三维速度动作版, 对齐已验证 KillChain 环境).

设计理念 (与方式 B 完全相反):
  - 让 RL policy 看到的环境与 Python KillChain 训练时**完全一致**
  - AFSIM 作为物理/传感器/通信承载层运行 (位置同步 + 真实 WsfTrackMessage 通信 + 武器命令发射)
  - AFSIM 的 5001 TrackEvent 被 Python 缓存, S 节点 strategy=1 基于真实 track 生成 detection_report
  - AFSIM 的 5003 MessageReceived 驱动 D/I buffer 入库, Python 不再用内部 message_queue 模拟通信到达
  - 5002 STRIKE_RESULT 仅用于清理 active_engagements 表 (不影响 RL state/reward)

继承关系:
  - 继承 KillChainEnv, 复用 step 主流程、_process_actions、_process_arrived_messages、
    _execute_kill_chain、get_obs、get_share_obs、get_available_actions
  - 仅 override 与 AFSIM 强相关的部分:
      1. __init__ / reset  : 用 AFSIM Bridge + NodeMapper 替代 _initialize_nodes 的随机撒点
      2. step              : 兼容 Tuple 混合动作与算法侧 flatten 后的 5 维动作
      3. _process_movement : 将连续 vx/vy/vz 方向转换为固定速度位移, 仍通过原 EntityMoveTo 下发
      4. _process_actions    : S 用 AFSIM track 生成情报; strategy=2 下发 SendTrackMessage 而不是 message_queue
      5. _process_arrived_messages : 消费 AFSIM 5003 并把业务消息放入 D/I 或中继 tx_buffer
      6. _perform_strike_logic : 在父类 pk-based 决策的同时, 向 AFSIM 发 FireWeapon
      7. _execute_kill_chain   : 在父类逻辑前 drain 5002 事件用于日志和 engagement 清理

约束:
  - n_rollout_threads 必须 = 1 (单 AFSIM 实例)
  - 动作空间: Tuple(MultiDiscrete([3, 5]), Box(3,))，算法侧 flatten 为 [strategy, candidate_idx, vx, vy, vz]
  - 观测空间: 对齐连续运动版 KillChain，self state 增加 vx/vy/vz
  - cfg.base_strike_range = 1.5 km (与 SAMAPPO_Ex10_10_4_PLUS 训练时对齐)
"""
import gym
import numpy as np
import logging
import time
import math
from gym import spaces
from typing import List, Tuple, Dict, Any, Optional

from onpolicy.envs.kill_chain.KillChain_Env import KillChainEnv
from onpolicy.envs.kill_chain.core import (
    NodeType, TaskStage, TaskEntity, Message, Interaction,
)

from onpolicy.envs.afsim_kill_chain.config import Config as AFSIMConfig_Cls
from onpolicy.envs.afsim_kill_chain.node_mapping import NodeMapper
from onpolicy.envs.afsim_kill_chain.afsim_bridge import AFSIMBridge


class AFSIMKillChainEnv(KillChainEnv):
    """
    AFSIM 适配层环境.

    对 RL policy 而言, 它看到的状态/动作/奖励与原 KillChainEnv 完全相同.
    底层差异:
      - 节点位置不再是 _initialize_nodes 随机撒点, 而是从 AFSIM DDS 1004 同步
      - 连续速度动作先在 Python 侧积分为下一步目标位置，再通过 DDS 9001 EntityMoveTo 下发到 AFSIM
      - I 开火时除了父类的 pk-based 决策, 还会发 9001 FireWeapon 让 AFSIM 物理执行
    """

    def __init__(self, args, rank=0, eval_mode=False):
        # 不调 super().__init__(), 因为父类会用 KillChain Config (kill_chain.config.Config)
        # 我们要用 afsim_kill_chain.config.Config (兼容父类接口, 多了 afsim 子配置)
        gym.Env.__init__(self)

        self.rank = rank
        self.args = args
        self.eval_mode = eval_mode

        # AFSIM 专用 config：保留通信/打击接口，运动学对齐连续三维速度版 KillChain
        self.cfg = AFSIMConfig_Cls()
        self.logger = logging.getLogger(f"AFSIMKillChainEnv[r{rank}]")

        # ===== 几何参数 =====
        self.map_size = np.array([
            self.cfg.space_size_x,
            self.cfg.space_size_y,
            self.cfg.space_size_z,
        ], dtype=np.float32)
        self.max_span = max(self.cfg.space_size_x, self.cfg.space_size_y, self.cfg.space_size_z)

        # ===== 智能体规模 =====
        self.n_agents = self.cfg.n_agents
        self.n_total = self.cfg.n_total_nodes
        self.max_steps = self.args.episode_length
        self.current_step = 0

        self.max_agents_limit = self.cfg.max_agents_limit
        self.max_targets_limit = self.cfg.max_targets_limit

        self.curr_n_target = self.cfg.n_target
        self.prev_completion_ratio = 0.0

        # ===== 容器 =====
        self.nodes = []
        self.tasks = []
        self.interaction_history = []
        self.message_queue = []
        self.msg_counter = 0

        # ===== 动作空间: 与连续运动版 KillChain 对齐 =====
        # 算法侧已经支持 Tuple 混合动作空间，最终传入 env.step 的动作通常会被 flatten 为:
        # [strategy, candidate_idx, vx, vy, vz]
        self.max_candidates = 5
        self.task_feat_dim = 7
        self.stats_feat_dim = getattr(self.cfg, "stats_feat_dim", 6)

        self.action_space = []
        for _ in range(self.n_agents):
            act_space = spaces.Tuple((
                spaces.MultiDiscrete([3, self.max_candidates]),
                spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            ))
            self.action_space.append(act_space)

        # ===== 观测空间: self state 增加当前速度 vx/vy/vz =====
        self.buffer_feat_dim = 7
        self.compass_dim = 6
        self.self_feat_dim = 13
        self.max_observed_tasks = self.max_targets_limit

        self.obs_dim = (
            self.self_feat_dim +
            self.buffer_feat_dim +
            self.compass_dim +
            self.stats_feat_dim +
            (self.task_feat_dim * self.max_observed_tasks) +
            (self.max_candidates * 7)
        )

        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # share obs: 每个 agent 为 [x,y,z,type,alive,vx,vy,vz] 共 8 维
        self.agent_share_feat_dim = 8
        self.share_obs_dim = (
            (self.max_agents_limit * self.agent_share_feat_dim)
            + self.stats_feat_dim
            + (self.max_targets_limit * 5)
        )
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # ===== 统计字段 =====
        self.total_reward = 0.0
        self.cooperation_count = 0.0
        self.completed_tasks = 0.0

        # ===== AFSIM Bridge + NodeMapper =====
        self._mock_mode = getattr(args, "afsim_mock_mode", False)
        self.mapper = NodeMapper(self.cfg)
        self.bridge = AFSIMBridge(
            self.cfg, self.mapper,
            mock_mode=self._mock_mode,
            logger=self.logger,
        )
        self._bridge_started = False

        # ===== Engagement 表: track_id_str <-> task_id =====
        # I 节点开火时, 我们用 "py_<task_id>_<i_node>" 当虚拟 track_id 发给 AFSIM
        # 5002 回来时按 track_id 清理这两张表
        self.active_engagements: Dict[str, str] = {}   # track_id_str -> task_id
        self.task_to_track: Dict[str, str] = {}        # task_id -> track_id_str

        # ===== AFSIM 通信承载状态 =====
        # Python 业务消息不能塞进 WsfTrackMessage；这里保存 msg_id/task_id/type 等语义上下文。
        # AFSIM 只负责发送原生 WsfTrackMessage，5003 到达后再用该表恢复 Python 业务消息。
        self.pending_message_table: List[Dict[str, Any]] = []
        self.pending_message_timeout_steps = int(getattr(self.cfg.afsim, "pending_message_timeout_steps", 20))

        # episode 计数 (给 SimReset 生成稳定可复现的 seed)
        self.episode_count = 0

        # ===== reset 中填充的占位字段 =====
        self.target_nodes_list = []
        self.max_mission_count = 0
        self.spawned_tasks_count = 0
        self.episode_pks = []
        self.episode_time_effs = []
        self.episode_paper_effs = []
        self.detected_targets_history = set()
        self.occupied_positions = set()
        self.step_buffer_loads = []
        self.type_episode_rewards = {
            NodeType.SENSOR: 0.0,
            NodeType.DECISION: 0.0,
            NodeType.INTERCEPTOR: 0.0,
        }

    # ==================================================================
    # Python 侧 AFSIM 消息闭环日志
    # ==================================================================
    def _interaction_log_enabled(self) -> bool:
        return bool(getattr(self.cfg.afsim, "interaction_debug_log", True))

    def _clog(self, text: str, force: bool = False):
        """Communication log: 只打印 AFSIM/RL 消息闭环关键节点。"""
        if force or self._interaction_log_enabled():
            self.logger.info(text)

    @staticmethod
    def _ntype_name(node_type) -> str:
        try:
            return node_type.name
        except Exception:
            return str(node_type)

    def _buffer_summary(self) -> str:
        if not getattr(self.cfg.afsim, "log_pending_table_size", True):
            return ""
        try:
            total_tx = sum(len(getattr(n, "tx_buffer", [])) for n in self.nodes[:self.n_agents])
            total_data = sum(len(getattr(n, "data_buffer", [])) for n in self.nodes[:self.n_agents])
            return f" pending={len(self.pending_message_table)} total_tx={total_tx} total_data={total_data}"
        except Exception:
            return ""

    def _msg_brief(self, msg: Message) -> str:
        c = getattr(msg, "content", {}) or {}
        return (
            f"msg_id={getattr(msg, 'msg_id', '')} type={getattr(msg, 'msg_type', '')} "
            f"task={c.get('task_id', '')} target_node={c.get('target_id', '')} "
            f"target_platform={c.get('target_platform_id', c.get('target_platform_index', 0))} "
            f"track='{c.get('current_track_id_str') or c.get('track_id_str') or c.get('original_track_id_str') or ''}' "
            f"hop={getattr(msg, 'hop_count', 0)} path_prob={float(getattr(msg, 'path_prob', 1.0)):.3f}"
        )

    # ==================================================================
    # reset
    # ==================================================================
    def reset(self):
        """
        重置环境, 返回 (obs, share_obs, available_actions).

        与父类 KillChainEnv.reset 的差异:
          - 节点位置不再用 _initialize_nodes (随机撒点),
            而是用 NodeMapper 从 AFSIM 平台真实位置构造
          - 启动/软重置 AFSIM Bridge
          - 等 AFSIM 平台就绪后再构建 nodes
        """
        self._clog(f"[PY-RESET] begin episode={self.episode_count} bridge_started={self._bridge_started} mock={self._mock_mode}", force=True)

        # ===== Python 内部状态重置 (与父类一致) =====
        self.current_step = 0
        self.total_reward = 0.0
        self.cooperation_count = 0.0
        self.completed_tasks = 0.0
        self.episode_pks = []
        self.episode_time_effs = []
        self.episode_paper_effs = []
        self.detected_targets_history = set()
        self.occupied_positions = set()
        self.step_buffer_loads = []
        self.type_episode_rewards = {
            NodeType.SENSOR: 0.0,
            NodeType.DECISION: 0.0,
            NodeType.INTERCEPTOR: 0.0,
        }
        self.nodes = []
        self.tasks = []
        self.interaction_history = []
        self.message_queue = []
        self.msg_counter = 0
        self.prev_completion_ratio = 0.0

        # AFSIM 适配: 清空 engagement 映射
        self.active_engagements = {}
        self.task_to_track = {}
        self.pending_message_table = []

        # ===== AFSIM Bridge 启动 / 硬重启 =====
        if not self._bridge_started:
            ok = self.bridge.start()
            if not ok:
                raise RuntimeError("AFSIMBridge.start() failed")
            self._bridge_started = True
            self._clog("[PY-RESET] DDS bridge started; subscribed to 1002/1003/1004/5001/5002/5003", force=True)
        else:
            self.episode_count += 1
            # [AFSIM reset owner]
            # 这里只在真正调用 env.reset() 时发送一次 SimRestart。
            # Runner 不再在每个 episode 开头额外 reset；VecEnv 在上一回合 done=True 时会自动调用一次 reset。
            # 这样避免“done 自动 reset 一次 + 下一 episode 开头又显式 reset 一次”导致 AFSIM 连续重启两次。
            if not self._mock_mode:
                self._send_sim_restart()

                # SimRestart 是异步 RequestReset：AFSIM 会完成当前仿真步后再重载场景。
                # 第一遍清理：丢弃上一回合本地缓存，避免旧平台状态直接让 wait_for_platforms_ready 提前返回。
                self.bridge.clear_runtime_state(clear_mapper=True, clear_origin=False)

                restart_sleep = float(getattr(self.cfg.afsim, "restart_post_sleep_sec", 3.0))
                if restart_sleep > 0.0:
                    self.logger.info(f"[Bridge] waiting {restart_sleep:.1f}s for AFSIM SimRestart to settle")
                    time.sleep(restart_sleep)

                # 第二遍清理：丢弃 RequestReset 过程中 DDS 后台线程收到的残留旧状态。
                # 后面的 wait_for_platforms_ready 只接受重启完成后的新 PlatformAdded/MoveUpdate。
                self.bridge.clear_runtime_state(clear_mapper=True, clear_origin=False)
            else:
                # mock 模式没有 AFSIM，只清 Python 本地状态。
                self.bridge.soft_reset(randomize=False)

        # ===== 等平台就绪 =====
        if not self._mock_mode:
            ready, reason = self.bridge.wait_for_platforms_ready(
                timeout_sec=self.cfg.afsim.reset_timeout_sec
            )
            if not ready:
                raise RuntimeError(f"Platforms not ready after reset: {reason}")
            time.sleep(self.cfg.afsim.reset_wait_sec)

        # ===== 决定 curr_n_target (照抄父类逻辑) =====
        use_random_n_target = getattr(self.cfg, "use_random_n_target", False)
        train_n_target_min = getattr(self.cfg, "train_n_target_min", self.cfg.n_target)
        train_n_target_max = getattr(self.cfg, "train_n_target_max", self.cfg.n_target)
        eval_n_target_list = getattr(self.cfg, "eval_n_target_list", [self.cfg.n_target])

        if self.eval_mode:
            if len(eval_n_target_list) == 0:
                self.curr_n_target = self.cfg.n_target
            else:
                self.curr_n_target = int(eval_n_target_list[self.rank % len(eval_n_target_list)])
        else:
            if use_random_n_target:
                low = max(1, int(train_n_target_min))
                high = min(int(train_n_target_max), int(self.max_targets_limit))
                if high < low:
                    high = low
                self.curr_n_target = int(np.random.randint(low, high + 1))
            else:
                self.curr_n_target = min(int(self.cfg.n_target), int(self.max_targets_limit))

        # ===== 用 NodeMapper 从 AFSIM 真实位置构造节点 =====
        self.nodes = self.mapper.create_endogenous_nodes()
        self._sync_positions_from_bridge()

        target_nodes = self.mapper.create_target_nodes()
        if len(target_nodes) == 0 and self._mock_mode:
            target_nodes = self._spawn_mock_targets(
                n_targets=min(self.curr_n_target, self.cfg.max_targets_limit)
            )

        self.nodes.extend(target_nodes)
        self.target_nodes_list = target_nodes
        self.curr_n_target = len(target_nodes)
        self.max_mission_count = self.curr_n_target
        self.spawned_tasks_count = 0
        self.completed_tasks = 0.0

        # ===== 任务初始化 (与父类一致) =====
        self.tasks = []
        for t_node in target_nodes:
            task = TaskEntity(
                task_id=f"Task-0-{t_node.local_id}",
                target_node_id=t_node.node_id,
                time_window=self.cfg.task_duration,
                total_time=self.cfg.task_duration,
                start_time=0.0,
            )
            task.stage = TaskStage.PENDING
            self.tasks.append(task)

        # ===== 父类 obs/share_obs/avail_actions (sanitize 包装见末尾) =====
        obs = self.get_obs()
        share_obs = self.get_share_obs(obs)
        available_actions = self.get_available_actions()

        try:
            role_counts = {}
            for n in self.nodes[:self.n_agents]:
                role_counts[self._ntype_name(n.node_type)] = role_counts.get(self._ntype_name(n.node_type), 0) + 1
            self._clog(
                f"[PY-RESET] ready: agents={self.n_agents} targets={self.curr_n_target} "
                f"roles={role_counts} tasks={len(self.tasks)}" + self._buffer_summary(),
                force=True,
            )
        except Exception:
            pass

        return obs, share_obs, available_actions

    # ==================================================================
    # step/action: 兼容 Tuple 混合动作与 flatten 后 5 维动作
    # ==================================================================
    @staticmethod
    def _flatten_one_action(action):
        """
        将动作统一成 [strategy, candidate_idx, vx, vy, vz]。

        MAPPO 的混合动作头通常会输出 flatten 后的 5 维向量；
        但手工测试 env.action_space.sample() 时会得到 (MultiDiscrete, Box) 的 Tuple，
        这里做一层兼容，避免父类 _process_actions 读取 action[0]/action[1] 时出错。
        """
        if isinstance(action, (tuple, list)) and len(action) == 2:
            discrete = np.asarray(action[0], dtype=np.float32).reshape(-1)
            continuous = np.asarray(action[1], dtype=np.float32).reshape(-1)
            if discrete.size >= 2 and continuous.size >= 3:
                return np.concatenate([discrete[:2], continuous[:3]]).astype(np.float32)

        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size >= 5:
            return arr[:5].astype(np.float32)
        if arr.size == 3:
            # 兼容旧 [strategy, candidate_idx, move] 输入：移动维不再使用，速度给 0。
            return np.array([arr[0], arr[1], 0.0, 0.0, 0.0], dtype=np.float32)
        if arr.size == 2:
            return np.array([arr[0], arr[1], 0.0, 0.0, 0.0], dtype=np.float32)
        out = np.zeros(5, dtype=np.float32)
        out[:min(arr.size, 5)] = arr[:min(arr.size, 5)]
        return out

    def _normalize_actions(self, actions):
        return np.asarray([self._flatten_one_action(a) for a in actions], dtype=np.float32)

    def step(self, actions):
        actions = self._normalize_actions(actions)
        return super().step(actions)

    # ==================================================================
    # _process_movement: 连续 vx/vy/vz -> EntityMoveTo
    # ==================================================================
    def _process_movement(self, actions):
        """
        AFSIM 交互版连续三维速度动作处理。

        输入动作统一为:
            [strategy, candidate_idx, vx, vy, vz]

        其中 vx/vy/vz 不是直接修改 DDS 消息格式，而是在 Python 侧解释为“期望速度方向”:
        1. 裁剪到 [-velocity_action_clip, velocity_action_clip]；
        2. 归一化为三维期望航向；
        3. 按节点类型固定速度模长，例如当前 S/D/I 默认均为 50 m/s；
        4. 按最大转弯半径限制本 step 航向变化；
        5. 将 old_pos + actual_vel 作为下一步目标位置，通过原 EntityMoveTo 下发。

        注意: DDS/AFSIM 消息格式保持不变，仍是原 9001 JSON:
            {"MsgType":"EntityMoveTo", "PlatformId":..., "Location":[lat,lon,alt]}
        """
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        actions = self._normalize_actions(actions)

        agent_indices = np.arange(self.n_agents)
        np.random.shuffle(agent_indices)

        # 固定训练区域边界：zone1 = 10 km × 10 km × 4 km。
        # Python 局部 ENU 原点固定为 zone1 中心，因此 x/y 为 [-5, 5] km，z 为 [0, 4] km。
        # 不再根据第一个平台动态切换原点，也不再使用 slack 放宽边界。
        map_min = np.asarray(
            getattr(self.cfg, "afsim_motion_min_km", (-5.0, -5.0, 0.0)),
            dtype=np.float32,
        )
        map_max = np.asarray(
            getattr(self.cfg, "afsim_motion_max_km", (5.0, 5.0, 4.0)),
            dtype=np.float32,
        )

        action_clip = float(getattr(self.cfg, "velocity_action_clip", 1.0))
        commands: Dict[int, np.ndarray] = {}
        planned_vel: Dict[int, np.ndarray] = {}
        old_positions: Dict[int, np.ndarray] = {}

        for i in agent_indices:
            agent = self.nodes[i]
            if not agent.alive:
                continue

            desired_cmd = np.asarray(actions[i][2:5], dtype=np.float32)
            desired_cmd = np.clip(desired_cmd, -action_clip, action_clip)

            speed_km_per_step = self._get_node_speed_km_per_step(agent)
            cmd_norm = np.linalg.norm(desired_cmd)
            if cmd_norm > 1e-8:
                desired_dir = desired_cmd / cmd_norm
                desired_vel = desired_dir * speed_km_per_step
            else:
                # 没有有效方向时保持当前航向；若当前也无航向，则默认沿 +X 建立初始航向。
                if hasattr(agent, "vel") and np.linalg.norm(agent.vel) > 1e-8:
                    desired_vel = agent.vel / np.linalg.norm(agent.vel) * speed_km_per_step
                else:
                    desired_vel = np.array([speed_km_per_step, 0.0, 0.0], dtype=np.float32)

            current_vel = getattr(agent, "vel", np.zeros(3, dtype=np.float32))
            actual_vel = self._limit_turn_by_radius(
                current_vel=current_vel,
                desired_vel=desired_vel,
                speed_km_per_step=speed_km_per_step,
            )

            old_pos = agent.pos.copy()
            target_pos = old_pos + actual_vel

            out_of_boundary = np.any(target_pos < map_min) or np.any(target_pos > map_max)
            if out_of_boundary:
                target_pos = np.clip(target_pos, map_min, map_max)
                actual_vel = target_pos - old_pos
                rewards[i] -= 2.0
                agent.consume_energy(self.cfg.energy_cost_collision)
            else:
                agent.consume_energy(self.cfg.energy_cost_move)

            if not self._is_position_free(target_pos, self_uid=agent.node_id):
                target_pos = old_pos
                actual_vel = np.zeros(3, dtype=np.float32)
                rewards[i] -= 2.0
                agent.consume_energy(self.cfg.energy_cost_collision)

            commands[i] = target_pos.astype(np.float32)
            planned_vel[i] = actual_vel.astype(np.float32)
            old_positions[i] = old_pos

        if commands:
            sent = self.bridge.send_move_commands_batch(commands)
            if bool(getattr(self.cfg.afsim, "motion_debug_log", False)):
                every = max(1, int(getattr(self.cfg.afsim, "move_log_every_steps", 10)))
                if self.current_step % every == 0:
                    self._clog(f"[PY-MOVE] step={self.current_step} EntityMoveTo batch sent={sent}/{len(commands)}")

        # AFSIM 真实模式: sleep 一个 step 让物理推进；即使本步无移动，也要给 5001/5002/5003 事件时间。
        if not self._mock_mode:
            self.bridge.wait_step(self.cfg.afsim.step_wait_sec)
        else:
            # mock 模式: 直接把目标位置写到 snapshot 上，不改变消息格式，仅用于本地单元测试。
            for node_id, target in commands.items():
                snap = self.bridge._snapshots.get(node_id)
                if snap is not None:
                    snap.pos_km = target.copy()

        self._sync_positions_from_bridge()

        # 用 AFSIM 回传后的真实位置修正速度；若还没收到回传，则保留本步计划速度。
        for node_id, old_pos in old_positions.items():
            agent = self.nodes[node_id]
            synced_delta = agent.pos - old_pos
            if np.linalg.norm(synced_delta) > 1e-8:
                agent.update_velocity(synced_delta.astype(np.float32))
            else:
                agent.update_velocity(planned_vel[node_id])

        return rewards

    # ==================================================================
    # _perform_strike_logic: 父类 pk 决策 + 用 AFSIM 真实 track_id 发 FireWeapon
    # ==================================================================
    def _find_real_track_for_target(self, target_node_id: int):
        """
        在 AFSIM bridge 的 5001 track cache 里搜索一条指向 target_node_id 的真实 track.

        关键: AFSIM C++ 端的 FireWeapon 处理器**严格校验 track_id 格式**, 必须是
        AFSIM 自己生成的形式 (如 "sensor_3:7"), 否则 C++ 在解析阶段直接拒收并弹错误框
        ("Invalid TrackId format for ..."). 我们之前用 "py_<task_id>_<node>" 自创格式
        会被拒, 所以现在改成: fire 时去 bridge 的 track cache 里找这个 target 对应的
        真实 sensor track, 用它的 track_id_str 调 FireWeapon.

        注: 5001 数据在这里**仅用于查表, 不进入 obs/reward**. RL 决策仍由 Python
        父类多跳模拟驱动 (符合用户"5001/5003 不消费"的原意 — 不消费指不参与决策,
        但允许在 fire 这种"翻译为 AFSIM 命令"的边缘场景做查询).

        Returns:
            TrackSnapshot 或 None.  None 表示当前没有任何 sensor 在跟踪这个 target,
            此时不向 AFSIM 发 FireWeapon (Python pk 决策不变, AFSIM 物理层不执行).
        """
        # 遍历所有 SENSOR node, 找最先匹配上 target 的真实 track
        # n_sensor 通常是 5, max_candidates 也是 5, 一次查询很快
        n_sensor = self.cfg.n_sensor
        for s_node_id in range(n_sensor):
            try:
                tracks = self.bridge.get_tracks_for_sensor(s_node_id, include_stale=False)
            except Exception:
                continue
            for tr in tracks:
                if tr.target_node_id == target_node_id and tr.msg_type != "TrackRemoved":
                    return tr
        return None

    def _perform_strike_logic(self, i_node, msg):
        """
        AFSIM 集成版 strike 逻辑.

        设计:
          1. super()._perform_strike_logic 让父类完成所有 RL 相关计算:
             pk = p1 * p2 * p3 * p_link * p4
             - pk >= min_kill_probability (0.6) → task COMPLETED, reward 排队分发
             - 否则 → 写 soft_failure_penalty 到 pending_reward_distribution
             RL policy 看到的状态/奖励与 KillChain 训练时**完全相同**.

          2. 在父类决策之外, 用 AFSIM 真实 sensor track_id 发 FireWeapon.
             - 找到真 track → AFSIM 发射武器, 物理飞行, 最终回 5002 STRIKE_RESULT
             - 找不到真 track (没有 sensor 在跟) → 不发 FireWeapon, 仅日志.
               Python 的 pk 决策不受影响 (super 已调用).

          3. 5002 STRIKE_RESULT 在 _drain_strike_events 里只做清理 + 日志, 不参与 reward.

        代价: Python 与 AFSIM 的 "target 死活" 状态可能不一致, 但对推理验证场景没问题.
        """
        # ===== 1. 父类 RL 决策 (pk 计算, stage 设置, reward 排队) =====
        super()._perform_strike_logic(i_node, msg)

        # ===== 2. 用 AFSIM 真实 track_id 发 FireWeapon =====
        task_id = msg.content.get("task_id", None)
        target_id = msg.content.get("target_id", None)
        if task_id is None or target_id is None:
            return

        # 同一 task 已经在 engagement 里了 → 不重复发火
        if task_id in self.task_to_track:
            return

        target_node = next((n for n in self.nodes if n.node_id == target_id), None)
        if target_node is None or not target_node.alive:
            return

        # 关键修复: 必须用 AFSIM 真实的 track_id (来自 5001), 不能自创格式.
        # 如果当前没有任何 sensor 在跟踪这个 target, 跳过 AFSIM 开火.
        real_track = self._find_real_track_for_target(target_id)
        if real_track is None:
            self.logger.info(
                f"[StrikeFire] I{i_node.node_id} skip AFSIM fire: "
                f"no live sensor track for target_node={target_id} "
                f"task='{task_id}' (Python pk decision still applies)"
            )
            return

        real_track_id = real_track.track_id_str   # 如 "sensor_3:7"

        ok = self.bridge.send_fire_weapon(
            i_node.node_id,
            real_track_id,
            target_lat_deg=float(real_track.target_lat_deg),
            target_lon_deg=float(real_track.target_lon_deg),
            target_alt_m=float(real_track.target_alt_m),
            target_platform_index=int(real_track.target_platform_index),
        )
        if ok:
            self.active_engagements[real_track_id] = task_id
            self.task_to_track[task_id] = real_track_id
            i_node.last_known_track_id = real_track_id
            self.logger.info(
                f"[StrikeFire] I{i_node.node_id} → AFSIM FireWeapon "
                f"task='{task_id}' target_node={target_id} "
                f"track='{real_track_id}' (real AFSIM sensor track, Python pk-decision already made)"
            )

    # ==================================================================
    # _execute_kill_chain: 父类逻辑 + drain 5002 事件
    # ==================================================================
    def _execute_kill_chain(self) -> np.ndarray:
        """
        与父类逻辑一致, 但在前面 drain 一次 5002 事件 (仅清理 + 日志).

        rewards 完全由父类决定, 5002 不参与 reward 计算 (符合"AFSIM 只跑物理"原则).
        """
        # 1. drain AFSIM 5002 事件 (清理 engagement 表 + 日志记录)
        self._drain_strike_events()

        # 2. 父类执行所有 RL 相关逻辑 (pending_reward_distribution 分发, 时间惩罚, 超时)
        rewards = super()._execute_kill_chain()

        return rewards

    def _drain_strike_events(self):
        """
        消费 bridge 累积的 5002 STRIKE_RESULT 事件.

        本方案下 5002 不影响 RL state/reward, 仅:
          1. 清理 active_engagements / task_to_track (避免内存膨胀)
          2. 打印日志, 让用户看到 AFSIM 物理结算结果 (HIT/MISS/TERMINATED)
        """
        try:
            events = self.bridge.pop_strike_events()
        except Exception as e:
            self.logger.warning(f"[StrikeDrain] pop_strike_events failed: {e}")
            return

        for ev in events:
            track_id = ev.track_id_str or ""
            if not track_id:
                continue

            task_id = self.active_engagements.pop(track_id, None)
            if task_id is None:
                # 不在我们的 engagement 表里, 可能是上一 episode 残留, 或 AFSIM 自发的事件
                continue

            self.task_to_track.pop(task_id, None)

            et = ev.event_type
            if et == "WEAPON_HIT" and ev.target_killed:
                self.logger.info(
                    f"[AFSIM] HIT+KILLED track='{track_id}' task='{task_id}' "
                    f"shooter_node={ev.shooter_node_id}"
                )
            elif et == "WEAPON_HIT":
                self.logger.info(f"[AFSIM] HIT (not killed) track='{track_id}' task='{task_id}'")
            elif et == "WEAPON_MISS":
                self.logger.info(
                    f"[AFSIM] MISS track='{track_id}' task='{task_id}' "
                    f"miss_dist={ev.miss_distance_m:.0f}m"
                )
            elif et == "WEAPON_TERMINATED":
                self.logger.info(f"[AFSIM] TERMINATED track='{track_id}' task='{task_id}'")
            elif et == "WEAPON_FIRED":
                self.logger.debug(f"[AFSIM] FIRED track='{track_id}' task='{task_id}'")

    # ==================================================================
    # AFSIM comm closed-loop helpers: Python semantic msg <-> AFSIM WsfTrackMessage
    # ==================================================================
    @staticmethod
    def _extract_track_local_number(track_id_str: str, default: int = 0) -> int:
        """兼容 AFSIM 原生 'owner.local' 和历史桥接层 'owner:local' 两种字符串。"""
        if not track_id_str:
            return int(default)
        sep_pos = max(track_id_str.rfind('.'), track_id_str.rfind(':'))
        if sep_pos < 0 or sep_pos + 1 >= len(track_id_str):
            return int(default)
        try:
            return int(track_id_str[sep_pos + 1:])
        except Exception:
            return int(default)

    def _get_afsim_track_candidates_for_sensor(self, sensor_node_id: int):
        """返回某个 S 节点当前可用于生成 detection_report 的 AFSIM 真实 track。"""
        try:
            tracks = self.bridge.get_tracks_for_sensor(sensor_node_id, include_stale=False)
        except Exception as e:
            self.logger.debug(f"[AFSIM-TRACK] get_tracks_for_sensor({sensor_node_id}) failed: {e}")
            return []

        candidates = []
        seen_targets = set()
        for tr in tracks:
            if tr.msg_type == "TrackRemoved":
                continue
            if tr.target_node_id < self.n_agents:
                continue
            target_node = next((n for n in self.nodes if n.node_id == tr.target_node_id), None)
            if target_node is None or not target_node.alive:
                continue
            task = next((t for t in self.tasks
                         if t.target_node_id == target_node.node_id
                         and t.stage not in [TaskStage.COMPLETED, TaskStage.FAILED]), None)
            if task is None:
                continue

            # 同一个目标可能有多条 track，优先保留 range 更近 / quality 更高的那条。
            key = int(tr.target_platform_index) if tr.target_platform_index > 0 else int(target_node.node_id)
            sort_range = tr.range_m if tr.range_m >= 0 else 1e18
            score_tuple = (sort_range, -float(tr.quality))
            candidates.append((score_tuple, tr, target_node, task))
            seen_targets.add(key)

        candidates.sort(key=lambda item: item[0])

        # 输出元素: (track_snapshot, target_node, task)
        return [(tr, target_node, task) for _, tr, target_node, task in candidates]

    def _make_pending_key(self, src_node_id: int, next_hop_node_id: int,
                          target_platform_index: int, track_id_str: str) -> Tuple[int, int, int, str]:
        return (int(src_node_id), int(next_hop_node_id), int(target_platform_index), str(track_id_str or ""))

    def _register_pending_message(self, msg: Message, src_node_id: int, next_hop_node_id: int,
                                  track_id_str: str, target_platform_index: int):
        """登记一条已交给 AFSIM comm 承载、等待 5003 回执的 Python 业务消息。"""
        self.pending_message_table.append({
            "msg": msg,
            "src_node_id": int(src_node_id),
            "next_hop_node_id": int(next_hop_node_id),
            "target_platform_index": int(target_platform_index),
            "track_id_str": str(track_id_str or ""),
            "msg_type": msg.msg_type,
            "target_node_type": msg.target_node_type,
            "send_step": int(self.current_step),
            "key": self._make_pending_key(src_node_id, next_hop_node_id, target_platform_index, track_id_str),
        })
        self._clog(
            f"[PY-PENDING+] step={self.current_step} src={src_node_id} next={next_hop_node_id} "
            f"{self._msg_brief(msg)}" + self._buffer_summary()
        )

    def _prune_expired_pending_messages(self):
        if not hasattr(self, "pending_message_table"):
            self.pending_message_table = []
        timeout = int(getattr(self, "pending_message_timeout_steps", 20))
        if timeout <= 0:
            return
        keep = []
        dropped = 0
        for item in self.pending_message_table:
            if int(self.current_step) - int(item.get("send_step", self.current_step)) <= timeout:
                keep.append(item)
            else:
                dropped += 1
        if dropped > 0:
            self._clog(f"[PY-PENDING-EXPIRE] step={self.current_step} dropped={dropped} keep={len(keep)}")
        self.pending_message_table = keep

    def _match_pending_message(self, ev):
        """
        用 5003 事件匹配 Python pending_message_table。
        注意 WsfTrackMessage 不承载 Python msg_id，所以这里按 src/receiver/target/track 做软匹配；
        如果 AFSIM 接收端把 raw track 相关成本地 track，track_id 可能变化，因此 track_id 只作为优先条件。
        """
        if not hasattr(self, "pending_message_table"):
            self.pending_message_table = []

        best_idx = -1
        best_score = -1
        for idx, item in enumerate(self.pending_message_table):
            if int(item.get("next_hop_node_id", -999)) != int(ev.receiver_node_id):
                continue
            if int(item.get("src_node_id", -999)) != int(ev.sender_node_id):
                continue

            score = 10
            if int(item.get("target_platform_index", 0)) > 0 and int(ev.target_platform_index) > 0:
                if int(item.get("target_platform_index")) == int(ev.target_platform_index):
                    score += 5
                else:
                    continue

            pending_track = str(item.get("track_id_str", ""))
            arrived_track = str(ev.track_id_str or "")
            if pending_track and arrived_track:
                if pending_track == arrived_track:
                    score += 5
                elif self._extract_track_local_number(pending_track) == int(ev.track_local_number):
                    score += 1

            # 越早发出的越优先匹配，避免同一链路同一目标连续消息乱序误配。
            score -= max(0, int(self.current_step) - int(item.get("send_step", self.current_step))) * 0.01

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            return None
        return self.pending_message_table.pop(best_idx)

    def _drain_message_arrivals(self, rewards: np.ndarray):
        """消费 AFSIM 5003，把已真实到达的 WsfTrackMessage 恢复成 Python 业务消息。"""
        self._prune_expired_pending_messages()
        try:
            arrivals = self.bridge.pop_message_arrivals()
        except Exception as e:
            self.logger.warning(f"[AFSIM-COMM] pop_message_arrivals failed: {e}")
            return

        if arrivals:
            self._clog(f"[PY-5003-DRAIN] step={self.current_step} arrivals={len(arrivals)}" + self._buffer_summary())

        for ev in arrivals:
            receiver = next((n for n in self.nodes if n.node_id == ev.receiver_node_id), None)
            if receiver is None or not receiver.alive:
                continue

            pending = self._match_pending_message(ev)
            if pending is None:
                self._clog(
                    f"[PY-5003-UNMATCHED] step={self.current_step} receiver={ev.receiver_node_id} "
                    f"sender={ev.sender_node_id} track='{ev.track_id_str}' "
                    f"local={ev.track_local_number} target_platform={ev.target_platform_index}" + self._buffer_summary()
                )
                continue

            msg: Message = pending["msg"]
            msg.to_id = receiver.node_id
            msg.arrival_time = self.current_step

            # 5003 中的 track 是 AFSIM comm 实际到达的 track，后续继续转发 / FireWeapon 优先使用它。
            if ev.track_id_str:
                msg.content["current_track_id_str"] = ev.track_id_str
                msg.content["track_id_str"] = ev.track_id_str
                msg.content["track_local_number"] = int(ev.track_local_number)
            if ev.target_platform_index > 0:
                msg.content["target_platform_id"] = int(ev.target_platform_index)
                msg.content["target_platform_index"] = int(ev.target_platform_index)
            if ev.target_lat_deg != 0.0 or ev.target_lon_deg != 0.0:
                msg.content["target_lat"] = float(ev.target_lat_deg)
                msg.content["target_lon"] = float(ev.target_lon_deg)
                msg.content["target_alt"] = float(ev.target_alt_m)
            msg.content["track_quality"] = float(ev.quality)

            self._clog(
                f"[PY-5003-MATCH] step={self.current_step} recv={receiver.node_id}/{self._ntype_name(receiver.node_type)} "
                f"from={ev.sender_node_id} track='{ev.track_id_str}' target_platform={ev.target_platform_index} "
                f"-> {self._msg_brief(msg)}" + self._buffer_summary()
            )

            # === 到达目标类型：进入业务 buffer ===
            if receiver.node_type == msg.target_node_type:
                final_success_reward = max(
                    0.0,
                    float(self.cfg.reward_routing_success) - float(msg.hop_count) * 0.1,
                )

                origin_id = int(msg.initial_source_id)
                if 0 <= origin_id < self.n_agents:
                    rewards[origin_id] += final_success_reward * 0.5

                last_hop_id = int(msg.from_id)
                if 0 <= last_hop_id < self.n_agents:
                    rewards[last_hop_id] += final_success_reward

                routing_reward = float(self.cfg.reward_routing_success) * 0.5
                for hop_node_id in msg.route_trace:
                    if 0 <= hop_node_id < self.n_agents and hop_node_id != last_hop_id:
                        rewards[hop_node_id] += routing_reward

                if receiver.node_type == NodeType.DECISION:
                    intel_data = dict(msg.content)
                    task_id = intel_data["task_id"]
                    intel_data["accumulated_comm_prob"] = float(msg.path_prob)
                    intel_data["s_d_trace"] = list(msg.route_trace) + [receiver.node_id]

                    current_quality = float(intel_data.get("confidence", 0.0)) * float(msg.path_prob)
                    existing_idx = -1
                    for idx, item in enumerate(receiver.data_buffer):
                        if item.get("task_id") == task_id:
                            existing_idx = idx
                            break
                    if existing_idx >= 0:
                        old_item = receiver.data_buffer[existing_idx]
                        old_quality = float(old_item.get("confidence", 0.0)) * float(old_item.get("accumulated_comm_prob", 1.0))
                        if current_quality > old_quality:
                            receiver.data_buffer[existing_idx] = intel_data
                            rewards[receiver.node_id] += float(self.cfg.reward_routing_success) * 0.2
                            self._clog(
                                f"[PY-BUFFER-D-UPDATE] step={self.current_step} D{receiver.node_id} "
                                f"task={task_id} quality={current_quality:.3f}>{old_quality:.3f} size={len(receiver.data_buffer)}"
                            )
                    else:
                        if len(receiver.data_buffer) < self.cfg.max_channels_D:
                            receiver.data_buffer.append(intel_data)
                            self._clog(
                                f"[PY-BUFFER-D-IN] step={self.current_step} D{receiver.node_id} "
                                f"task={task_id} from_msg={msg.msg_id} size={len(receiver.data_buffer)} "
                                f"track='{intel_data.get('track_id_str', '')}'"
                            )

                elif receiver.node_type == NodeType.INTERCEPTOR:
                    task_id = msg.content["task_id"]
                    prob_s_d = float(msg.content.get("accumulated_comm_prob_s_d", 1.0))
                    prob_q3 = float(msg.content.get("q3_prob", 1.0))
                    prob_d_i = float(msg.path_prob)
                    current_chain_quality = prob_s_d * prob_q3 * prob_d_i

                    existing_idx = -1
                    for idx, item in enumerate(receiver.data_buffer):
                        if isinstance(item, Message) and item.content.get("task_id") == task_id:
                            existing_idx = idx
                            break
                    if existing_idx >= 0:
                        old_msg = receiver.data_buffer[existing_idx]
                        old_chain_quality = (
                            float(old_msg.content.get("accumulated_comm_prob_s_d", 1.0)) *
                            float(old_msg.content.get("q3_prob", 1.0)) *
                            float(old_msg.path_prob)
                        )
                        if current_chain_quality > old_chain_quality:
                            receiver.data_buffer[existing_idx] = msg
                            self._clog(
                                f"[PY-BUFFER-I-UPDATE] step={self.current_step} I{receiver.node_id} "
                                f"task={task_id} chain={current_chain_quality:.3f}>{old_chain_quality:.3f} size={len(receiver.data_buffer)}"
                            )
                    else:
                        if len(receiver.data_buffer) < self.cfg.max_channels_I:
                            receiver.data_buffer.append(msg)
                            self._clog(
                                f"[PY-BUFFER-I-IN] step={self.current_step} I{receiver.node_id} "
                                f"task={task_id} from_msg={msg.msg_id} size={len(receiver.data_buffer)} "
                                f"track='{msg.content.get('track_id_str', '')}'"
                            )

            # === 还没到目标类型：交给中继节点 tx_buffer，等待下一次 strategy=2 ===
            else:
                if msg.hop_count < 10 and len(receiver.tx_buffer) < 20:
                    receiver.tx_buffer.append(msg)
                    self._clog(
                        f"[PY-BUFFER-RELAY-IN] step={self.current_step} node={receiver.node_id}/{self._ntype_name(receiver.node_type)} "
                        f"queued {self._msg_brief(msg)} tx_size={len(receiver.tx_buffer)}"
                    )

    # ==================================================================
    # _process_actions: S 基于 5001 track 产情报；通信由 AFSIM comm 承载
    # ==================================================================
    def _process_actions(self, actions: List[np.ndarray], rewards: np.ndarray):
        actions = self._normalize_actions(actions)

        # 先消费上一仿真步已经到达的 5003，使 D/I 在当前 step 就能看到 buffer。
        self._drain_message_arrivals(rewards)

        if self._interaction_log_enabled():
            active = [(i, int(a[0]), int(a[1])) for i, a in enumerate(actions) if int(a[0]) != 0]
            if active:
                self._clog(f"[PY-ACTION] step={self.current_step} active_actions={active[:12]}" + self._buffer_summary())

        for agent_id, action in enumerate(actions):
            agent = self.nodes[agent_id]
            if not agent.alive:
                continue

            strategy = int(action[0])
            candidate_idx = int(action[1])
            _, a_comm, _ = agent.get_actual_ranges(self.cfg)

            if strategy == 0:
                continue

            # ===============================================================
            # Strategy 1: Produce / Execute
            # ===============================================================
            if strategy == 1:
                # ---------------- SENSOR: 只能基于 AFSIM 5001 track 生成 detection_report ----------------
                if agent.node_type == NodeType.SENSOR:
                    track_candidates = self._get_afsim_track_candidates_for_sensor(agent.node_id)
                    if candidate_idx >= len(track_candidates) or candidate_idx >= self.max_candidates:
                        self._clog(
                            f"[PY-S-DETECT-SKIP] step={self.current_step} S{agent.node_id} "
                            f"candidate_idx={candidate_idx} available_tracks={len(track_candidates)}"
                        )
                        continue

                    track_snap, target_node, target_task = track_candidates[candidate_idx]
                    target_dist = float(np.linalg.norm(agent.pos - target_node.pos))

                    if target_node.node_id not in self.detected_targets_history:
                        rewards[agent_id] += self.cfg.detection_reward
                        self.detected_targets_history.add(target_node.node_id)
                    else:
                        rewards[agent_id] += 0.05

                    self.interaction_history.append(Interaction(
                        agent_id, target_node.node_id,
                        NodeType.SENSOR, NodeType.TARGET,
                        target_dist, "detect", None
                    ))
                    agent.consume_energy(self.cfg.energy_cost_detection)

                    if target_task.stage == TaskStage.PENDING:
                        target_task.stage = TaskStage.DETECTED
                        target_task.first_detection_time = float(self.current_step)

                    task_id = target_task.task_id
                    if len(agent.tx_buffer) >= 20:
                        continue

                    has_duplicate = any(
                        pending_msg.msg_type == "detection_report" and
                        pending_msg.content.get("task_id") == task_id
                        for pending_msg in agent.tx_buffer
                    )
                    if has_duplicate:
                        continue

                    q1 = self._calc_q1_detect_prob(agent, target_node)
                    track_local_number = self._extract_track_local_number(track_snap.track_id_str)
                    msg = Message(
                        msg_id=f"intel_{self.msg_counter}",
                        initial_source_id=agent_id,
                        from_id=agent_id,
                        to_id=-1,
                        target_node_type=NodeType.DECISION,
                        msg_type="detection_report",
                        content={
                            "task_id": task_id,
                            "confidence": q1,
                            "s_pos": agent.pos.copy(),
                            "t_pos": target_node.pos.copy(),
                            "s_reporter_id": agent_id,
                            "target_id": target_node.node_id,
                            # AFSIM track 引用；不进入 WsfTrackMessage，只供 Python 发转发指令/FireWeapon 使用。
                            "original_track_id_str": track_snap.track_id_str,
                            "current_track_id_str": track_snap.track_id_str,
                            "track_id_str": track_snap.track_id_str,
                            "track_local_number": track_local_number,
                            "target_platform_id": int(track_snap.target_platform_index),
                            "target_platform_index": int(track_snap.target_platform_index),
                            "target_lat": float(track_snap.target_lat_deg),
                            "target_lon": float(track_snap.target_lon_deg),
                            "target_alt": float(track_snap.target_alt_m),
                            "track_quality": float(track_snap.quality),
                        },
                        send_time=0,
                        arrival_time=0,
                        path_prob=1.0,
                    )
                    agent.tx_buffer.append(msg)
                    self.msg_counter += 1
                    self._clog(
                        f"[PY-S-DETECT] step={self.current_step} S{agent.node_id} -> tx_buffer "
                        f"{self._msg_brief(msg)} q1={q1:.3f} range={target_dist:.3f}km "
                        f"track_q={float(track_snap.quality):.2f} tx_size={len(agent.tx_buffer)}"
                    )

                # ---------------- DECISION: 生成 strike_command，继承 detection_report 的 track 引用 ----------------
                elif agent.node_type == NodeType.DECISION:
                    if len(agent.data_buffer) <= 0:
                        self._clog(f"[PY-D-CMD-SKIP] step={self.current_step} D{agent.node_id} data_buffer empty")
                        continue
                    best_idx = -1
                    best_score = -1.0
                    for idx, intel_data in enumerate(agent.data_buffer):
                        task_id = intel_data["task_id"]
                        task = next((t for t in self.tasks if t.task_id == task_id), None)
                        if task is None or task.stage in [TaskStage.COMPLETED, TaskStage.FAILED]:
                            continue
                        quality = float(intel_data.get("confidence", 0.0)) * float(intel_data.get("accumulated_comm_prob", 1.0))
                        urgency = 1.0 - (task.time_window / max(1.0, self.cfg.task_duration))
                        score = quality + 0.2 * urgency
                        if score > best_score:
                            best_score = score
                            best_idx = idx

                    if best_idx == -1:
                        continue
                    intel_data = agent.data_buffer.pop(best_idx)
                    task_id = intel_data["task_id"]
                    task = next((t for t in self.tasks if t.task_id == task_id), None)
                    if task is None or task.stage in [TaskStage.COMPLETED, TaskStage.FAILED]:
                        continue

                    has_cmd_in_buffer = any(
                        pending_msg.msg_type == "strike_command" and
                        pending_msg.content.get("task_id") == task_id
                        for pending_msg in agent.tx_buffer
                    )
                    if has_cmd_in_buffer:
                        continue

                    task.assigned_d = agent_id
                    task.stage = TaskStage.ASSIGNED
                    q3 = self._calc_q3_command_prob(agent, 1.0)

                    msg = Message(
                        msg_id=f"cmd_{self.msg_counter}",
                        initial_source_id=agent_id,
                        from_id=agent_id,
                        to_id=-1,
                        target_node_type=NodeType.INTERCEPTOR,
                        msg_type="strike_command",
                        content={
                            "task_id": task_id,
                            "target_id": task.target_node_id,
                            "origin_confidence": intel_data["confidence"],
                            "accumulated_comm_prob_s_d": intel_data.get("accumulated_comm_prob", 1.0),
                            "s_d_trace": intel_data.get("s_d_trace", []),
                            "q3_prob": q3,
                            "s_reporter_id": intel_data.get("s_reporter_id", -1),
                            # 继承 S→D 到达后确认的 AFSIM track 引用。
                            "original_track_id_str": intel_data.get("original_track_id_str", intel_data.get("track_id_str", "")),
                            "current_track_id_str": intel_data.get("current_track_id_str", intel_data.get("track_id_str", "")),
                            "track_id_str": intel_data.get("current_track_id_str", intel_data.get("track_id_str", "")),
                            "track_local_number": self._extract_track_local_number(
                                intel_data.get("current_track_id_str", intel_data.get("track_id_str", "")),
                                intel_data.get("track_local_number", 0),
                            ),
                            "target_platform_id": int(intel_data.get("target_platform_id", intel_data.get("target_platform_index", 0))),
                            "target_platform_index": int(intel_data.get("target_platform_index", intel_data.get("target_platform_id", 0))),
                            "target_lat": float(intel_data.get("target_lat", 0.0)),
                            "target_lon": float(intel_data.get("target_lon", 0.0)),
                            "target_alt": float(intel_data.get("target_alt", 0.0)),
                            "track_quality": float(intel_data.get("track_quality", 1.0)),
                        },
                        send_time=0,
                        arrival_time=0,
                        path_prob=1.0,
                    )
                    if len(agent.tx_buffer) < 20:
                        agent.tx_buffer.append(msg)
                        self.msg_counter += 1
                        rewards[agent_id] += self.cfg.assignment_reward
                        agent.consume_energy(self.cfg.energy_cost_comm)
                        self._clog(
                            f"[PY-D-CMD] step={self.current_step} D{agent.node_id} -> tx_buffer "
                            f"{self._msg_brief(msg)} q3={q3:.3f} tx_size={len(agent.tx_buffer)}"
                        )

                # ---------------- INTERCEPTOR ----------------
                elif agent.node_type == NodeType.INTERCEPTOR:
                    if len(agent.data_buffer) <= 0:
                        self._clog(f"[PY-I-FIRE-SKIP] step={self.current_step} I{agent.node_id} data_buffer empty")
                        continue
                    best_idx = -1
                    best_score = -1.0
                    for idx, msg in enumerate(agent.data_buffer):
                        if not isinstance(msg, Message):
                            continue
                        task_id = msg.content["task_id"]
                        task = next((t for t in self.tasks if t.task_id == task_id), None)
                        if task is None or task.stage in [TaskStage.COMPLETED, TaskStage.FAILED]:
                            continue
                        target_node = next((n for n in self.nodes if n.node_id == task.target_node_id), None)
                        if target_node is None or not target_node.alive:
                            continue
                        chain_quality = (
                            float(msg.content.get("accumulated_comm_prob_s_d", 1.0)) *
                            float(msg.content.get("q3_prob", 1.0)) *
                            float(msg.path_prob)
                        )
                        dist = np.linalg.norm(agent.pos - target_node.pos)
                        score = chain_quality + 0.2 * (1.0 / (1.0 + dist))
                        if score > best_score:
                            best_score = score
                            best_idx = idx

                    if best_idx != -1:
                        selected_msg = agent.data_buffer.pop(best_idx)
                        rewards[agent_id] += self.cfg.strike_initiation_reward
                        self._clog(
                            f"[PY-I-FIRE] step={self.current_step} I{agent.node_id} selected "
                            f"{self._msg_brief(selected_msg)} score={best_score:.3f}"
                        )
                        consumed = self._perform_strike_logic(agent, selected_msg)
                        if not consumed:
                            agent.data_buffer.insert(0, selected_msg)
                        agent.consume_energy(self.cfg.energy_cost_strike)

            # ===============================================================
            # Strategy 2: Comm / Relay via AFSIM WsfTrackMessage
            # ===============================================================
            elif strategy == 2:
                if len(agent.tx_buffer) <= 0:
                    self._clog(f"[PY-COMM-SKIP] step={self.current_step} node={agent.node_id}/{self._ntype_name(agent.node_type)} tx_buffer empty")
                    continue
                neighbor_candidates = []
                for other in self.nodes:
                    if other.node_id == agent.node_id:
                        continue
                    d = float(np.linalg.norm(agent.pos - other.pos))
                    link_prob = self._calc_q2_comm_prob(agent, other) if other.alive else 0.0
                    if other.alive and d <= a_comm and link_prob > 0.25:
                        neighbor_candidates.append((d, other, link_prob))
                neighbor_candidates.sort(key=lambda x: x[0])

                if candidate_idx >= len(neighbor_candidates) or candidate_idx >= self.max_candidates:
                    self._clog(
                        f"[PY-COMM-SKIP] step={self.current_step} node={agent.node_id}/{self._ntype_name(agent.node_type)} "
                        f"candidate_idx={candidate_idx} available_neighbors={len(neighbor_candidates)} tx_size={len(agent.tx_buffer)}"
                    )
                    continue

                target_dist, target_node, link_prob = neighbor_candidates[candidate_idx]
                msg = agent.tx_buffer.pop(0)

                if target_node.node_id in msg.route_trace:
                    self._clog(
                        f"[PY-COMM-LOOP-SKIP] step={self.current_step} node={agent.node_id} -> {target_node.node_id} "
                        f"route_trace={msg.route_trace} {self._msg_brief(msg)}"
                    )
                    agent.consume_energy(self.cfg.energy_cost_comm)
                    continue

                track_id_str = str(
                    msg.content.get("current_track_id_str") or
                    msg.content.get("track_id_str") or
                    msg.content.get("original_track_id_str") or
                    ""
                )
                track_local_number = self._extract_track_local_number(
                    track_id_str,
                    msg.content.get("track_local_number", 0),
                )
                target_platform_index = int(
                    msg.content.get("target_platform_id", msg.content.get("target_platform_index", 0))
                )

                # Python 仍然计算链路概率 / 时延 / 奖励；AFSIM 负责真实能不能收到。
                delay = self._calculate_comm_delay(agent, target_node, target_dist)
                msg.from_id = agent.node_id
                msg.to_id = target_node.node_id
                msg.path_prob *= float(link_prob)
                msg.hop_count += 1
                msg.route_trace.append(agent.node_id)
                msg.send_time = self.current_step
                msg.arrival_time = self.current_step + delay

                self._clog(
                    f"[PY-COMM-SEND] step={self.current_step} {agent.node_id}/{self._ntype_name(agent.node_type)} "
                    f"-> {target_node.node_id}/{self._ntype_name(target_node.node_type)} "
                    f"dist={target_dist:.3f}km q2={link_prob:.3f} delay={delay} "
                    f"{self._msg_brief(msg)}"
                )
                ok = self.bridge.send_track_message(
                    src_node_id=agent.node_id,
                    next_hop_node_id=target_node.node_id,
                    track_id_str=track_id_str,
                    track_local_number=track_local_number,
                    target_lat_deg=float(msg.content.get("target_lat", 0.0)),
                    target_lon_deg=float(msg.content.get("target_lon", 0.0)),
                    target_alt_m=float(msg.content.get("target_alt", 0.0)),
                    target_platform_index=target_platform_index,
                )
                if ok:
                    self._register_pending_message(
                        msg=msg,
                        src_node_id=agent.node_id,
                        next_hop_node_id=target_node.node_id,
                        track_id_str=track_id_str,
                        target_platform_index=target_platform_index,
                    )
                    agent.consume_energy(self.cfg.energy_cost_comm)
                    self.interaction_history.append(Interaction(
                        agent_id, target_node.node_id,
                        agent.node_type, target_node.node_type,
                        target_dist, "comm", None
                    ))
                else:
                    # 控制指令没发出去，消息放回队首，避免 Python 业务消息无故丢失。
                    agent.tx_buffer.insert(0, msg)
                    self._clog(
                        f"[PY-COMM-SEND-FAIL] step={self.current_step} restored_to_tx node={agent.node_id} "
                        f"{self._msg_brief(msg)} tx_size={len(agent.tx_buffer)}",
                        force=True,
                    )

    def _process_arrived_messages(self, rewards):
        """AFSIM 版不再消费 Python message_queue，只消费 AFSIM 5003。"""
        self.message_queue = []
        self._drain_message_arrivals(rewards)

    # ==================================================================
    # _perform_strike_logic: 父类 pk 决策 + 用到达 I 的 AFSIM track_id 发 FireWeapon
    # ==================================================================
    def _perform_strike_logic(self, i_node, msg):
        """
        AFSIM 集成版 strike 逻辑。
        I.data_buffer 中的 strike_command 已经来自 5003，content 里携带 current_track_id_str。
        Python 仍按父类计算 pk/reward，AFSIM 只执行物理 FireWeapon。
        """
        super()._perform_strike_logic(i_node, msg)

        task_id = msg.content.get("task_id", None)
        target_id = msg.content.get("target_id", None)
        if task_id is None or target_id is None:
            return True
        if task_id in self.task_to_track:
            return True

        target_node = next((n for n in self.nodes if n.node_id == target_id), None)
        if target_node is None or not target_node.alive:
            return True

        track_id = str(
            msg.content.get("current_track_id_str") or
            msg.content.get("track_id_str") or
            msg.content.get("original_track_id_str") or
            ""
        )

        # 兜底：如果业务消息没有携带 track，仍可从 5001 cache 找目标 track。
        fallback_track = None
        if not track_id:
            fallback_track = self._find_real_track_for_target(target_id)
            if fallback_track is not None:
                track_id = fallback_track.track_id_str

        if not track_id:
            self.logger.info(
                f"[StrikeFire] I{i_node.node_id} skip AFSIM fire: no track_id for "
                f"target_node={target_id} task='{task_id}' (Python pk decision still applies)"
            )
            return True

        target_lat = float(msg.content.get("target_lat", 0.0))
        target_lon = float(msg.content.get("target_lon", 0.0))
        target_alt = float(msg.content.get("target_alt", 0.0))
        target_platform_index = int(msg.content.get("target_platform_id", msg.content.get("target_platform_index", 0)))

        if fallback_track is not None:
            target_lat = float(fallback_track.target_lat_deg)
            target_lon = float(fallback_track.target_lon_deg)
            target_alt = float(fallback_track.target_alt_m)
            target_platform_index = int(fallback_track.target_platform_index)

        ok = self.bridge.send_fire_weapon(
            i_node.node_id,
            track_id,
            target_lat_deg=target_lat,
            target_lon_deg=target_lon,
            target_alt_m=target_alt,
            target_platform_index=target_platform_index,
        )
        if ok:
            self.active_engagements[track_id] = task_id
            self.task_to_track[task_id] = track_id
            i_node.last_known_track_id = track_id
            self.logger.info(
                f"[StrikeFire] I{i_node.node_id} → AFSIM FireWeapon "
                f"task='{task_id}' target_node={target_id} track='{track_id}'"
            )

        # True 表示该指令已被消费，不再塞回 I.data_buffer。
        return True

    # ==================================================================
    # _execute_kill_chain: drain 5002 + 父类奖励/超时逻辑
    # ==================================================================
    def _execute_kill_chain(self) -> np.ndarray:
        self._drain_strike_events()
        rewards = super()._execute_kill_chain()
        return rewards

    def get_available_actions(self):
        """AFSIM 版动作 mask：S 的 strategy=1 由 5001 track_cache 决定，strategy=2 仍由 Python 链路概率候选决定。"""
        avail_actions = []
        for i in range(self.n_agents):
            agent = self.nodes[i]
            mask = np.zeros(3 + self.max_candidates, dtype=np.float32)

            if not agent.alive:
                mask[0] = 1.0
                mask[3] = 1.0
                avail_actions.append(mask)
                continue

            mask[0] = 1.0
            candidate_mask = np.zeros(self.max_candidates, dtype=np.float32)

            if agent.node_type == NodeType.SENSOR:
                track_candidates = self._get_afsim_track_candidates_for_sensor(agent.node_id)
                valid_detect_slots = min(len(track_candidates), self.max_candidates)
                if valid_detect_slots > 0:
                    mask[1] = 1.0
                    candidate_mask[:valid_detect_slots] = 1.0
            elif agent.node_type in [NodeType.DECISION, NodeType.INTERCEPTOR]:
                if len(agent.data_buffer) > 0:
                    mask[1] = 1.0
                    candidate_mask[:] = 1.0

            if len(agent.tx_buffer) > 0:
                mask[2] = 1.0
                _, a_comm, _ = agent.get_actual_ranges(self.cfg)
                neighbor_candidates = []
                for other in self.nodes:
                    if other.node_id == agent.node_id:
                        continue
                    d = float(np.linalg.norm(agent.pos - other.pos))
                    link_prob = self._calc_q2_comm_prob(agent, other) if other.alive else 0.0
                    if other.alive and d <= a_comm and link_prob > 0.25:
                        neighbor_candidates.append((d, other))
                neighbor_candidates.sort(key=lambda x: x[0])
                valid_neighbor_slots = min(len(neighbor_candidates), self.max_candidates)
                candidate_mask[:valid_neighbor_slots] = 1.0

            if candidate_mask.sum() <= 0:
                candidate_mask[0] = 1.0
            mask[3:3 + self.max_candidates] = candidate_mask
            avail_actions.append(mask)

        return avail_actions

    # ==================================================================
    # SimRestart: AFSIM 原生硬重启 (替代不太可靠的 SimReset 软重置)
    # ==================================================================
    def _send_sim_restart(self) -> bool:
        """
        发 SimRestart 命令到 AFSIM, 调 AFSIM 原生 mSimPtr->RequestReset().

        与你 C++ 写的 SimReset (软重置) 的区别:
          - SimReset:  你自己实现的, 遍历每个平台 SetLocationLLA + GoToLocation + 清弹药.
                       理论上正确, 但实测 mover/task 状态遗留导致重置不完全.
          - SimRestart: AFSIM 原生 API, 等于 Warlock 的 "Reset Scenario" 按钮.
                        平台位置/simTime/mover/track/weapon/broken 全部回到场景文件初始值.

        SimRestart 是异步的: AFSIM 会"完成当前 step 然后 reset", 通常 1-3 秒内完成.
        Python 这边随后调 wait_for_platforms_ready 等 mapper 重新填满, 60s 超时基本足够.

        Returns:
            True 表示 DDS publish 成功 (不代表 AFSIM 一定成功 reset).
        """
        cmd = {"MsgType": "SimRestart"}
        ok = self.bridge._publish_control(cmd)
        if ok:
            self.logger.info("[Bridge] SimRestart sent (AFSIM native RequestReset)")
        else:
            self.logger.warning("[Bridge] SimRestart publish failed")
        return ok

    # ==================================================================
    # 辅助: 同步位置 / mock 兜底 / 关闭 / NaN sanitize
    # ==================================================================
    def _sync_positions_from_bridge(self):
        """
        从 bridge 快照同步所有节点 pos.

        NaN/inf 防御: AFSIM 极端状态下可能发出异常 LLA, 直接保留旧 pos 不更新,
        避免 NaN 传到 obs → policy logits NaN → FixedCategorical 崩溃.
        """
        positions = self.bridge.get_positions()
        alive_map = self.bridge.get_alive_map()

        for node in self.nodes:
            pos = positions.get(node.node_id)
            if pos is not None:
                if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                    self.logger.warning(
                        f"[SyncPos] node_id={node.node_id} got invalid pos={pos}, "
                        f"keep previous pos={node.pos}"
                    )
                else:
                    old_pos = node.pos.copy()
                    node.update_position_from_afsim(pos, timestamp=self.current_step)
                    delta = node.pos - old_pos
                    if hasattr(node, "update_velocity") and np.linalg.norm(delta) > 1e-8:
                        node.update_velocity(delta.astype(np.float32))
            alive = alive_map.get(node.node_id)
            if alive is not None:
                node.alive = alive

        # 兜底: 如果某节点 pos 仍是 NaN/inf (可能从未收到 MoveUpdate), 写零
        for node in self.nodes:
            if np.any(np.isnan(node.pos)) or np.any(np.isinf(node.pos)):
                self.logger.warning(
                    f"[SyncPos] node_id={node.node_id} has invalid pos={node.pos}, "
                    f"reset to [0,0,0]"
                )
                node.pos = np.zeros(3, dtype=np.float32)
                if hasattr(node, "update_velocity"):
                    node.update_velocity(np.zeros(3, dtype=np.float32))

    def _spawn_mock_targets(self, n_targets: int):
        """Mock 模式兜底: AFSIM 没发红方时自己造几个固定位置 TARGET (调试用)"""
        from onpolicy.envs.afsim_kill_chain.core import EndogenousNode
        targets = []
        for i in range(n_targets):
            node_id = self.n_agents + i
            bounds = np.array(self.cfg.world_bounds_km)
            rng = np.random.default_rng(seed=42 + self.rank * 1000 + i)
            pos = bounds * 0.3 + rng.random(3) * bounds * 0.4
            node = EndogenousNode(
                node_id=node_id,
                node_type=NodeType.TARGET,
                group_name="Target",
                local_id=i,
                energy=0.0,
                alive=True,
                pos=pos.astype(np.float32),
                raw_capability=np.zeros(4, dtype=np.float32),
                afsim_platform_index=-1,
                afsim_side="mock_red",
            )
            targets.append(node)
        return targets

    def close(self):
        if self._bridge_started:
            self.bridge.stop()
            self._bridge_started = False

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, mode='human'):
        pass

    # ==================================================================
    # NaN sanitize 包装: get_obs / get_share_obs
    # ==================================================================
    @staticmethod
    def _sanitize_array(arr):
        """把 list/np.ndarray 里的 NaN/inf 替换成有限值, 防止 policy logits 炸."""
        if isinstance(arr, list):
            return [AFSIMKillChainEnv._sanitize_array(x) for x in arr]
        if isinstance(arr, np.ndarray):
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(arr.dtype)
        return arr

    def get_obs(self):
        """父类 get_obs 出来过一遍 sanitize (防御 NaN)"""
        obs = super().get_obs()
        return self._sanitize_array(obs)

    def get_share_obs(self, obs=None):
        """父类 get_share_obs 出来过一遍 sanitize"""
        share_obs = super().get_share_obs(obs)
        return self._sanitize_array(share_obs)