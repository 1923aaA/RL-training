"""
杀伤链多智能体环境（完整版 - 通信时延模型）
"""
import gym
from gym import spaces
import numpy as np
import math
from typing import List, Tuple, Dict
from .config import Config
from .core import (
    EndogenousNode, TaskEntity, Interaction, Message,
    NodeType, TaskStage
)


class KillChainEnv(gym.Env):
    """
    杀伤链多智能体强化学习环境（通信时延版本）
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, args, rank=0, eval_mode=False):
        """初始化环境"""
        super(KillChainEnv, self).__init__()
        self.rank = rank
        self.args = args
        self.eval_mode = eval_mode
        self.cfg = Config()

        self.map_size = np.array([
            self.cfg.space_size_x,
            self.cfg.space_size_y,
            self.cfg.space_size_z
        ], dtype=np.float32)

        self.max_span = max(self.cfg.space_size_x, self.cfg.space_size_y, self.cfg.space_size_z)

        self.n_agents = self.cfg.n_agents
        self.n_total = self.cfg.n_total_nodes
        self.max_steps = self.args.episode_length
        self.current_step = 0

        self.max_agents_limit = self.cfg.max_agents_limit
        self.max_targets_limit = self.cfg.max_targets_limit

        # 当前回合真实目标数
        self.curr_n_target = self.cfg.n_target
        self.prev_completion_ratio = 0.0

        self.nodes: List[EndogenousNode] = []
        self.tasks: List[TaskEntity] = []
        self.interaction_history: List[Interaction] = []
        self.message_queue: List[Message] = []
        self.msg_counter = 0

        # 动作空间：混合动作 [strategy, candidate_idx, vx, vy, vz]
        # strategy/candidate_idx 为离散动作；vx/vy/vz 为连续动作，表示期望速度方向分量。
        self.max_candidates = 5
        self.task_feat_dim = 7
        self.stats_feat_dim = getattr(self.cfg, "stats_feat_dim", 6)

        self.action_space = []
        for _ in range(self.n_agents):
            act_space = spaces.Tuple((
                spaces.MultiDiscrete([3, self.max_candidates]),
                spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(3,),
                    dtype=np.float32
                )
            ))
            self.action_space.append(act_space)

        # 观测空间
        self.buffer_feat_dim = 7
        self.compass_dim = 6
        self.self_feat_dim = 13  # 原 10 维 + 当前速度 vx/vy/vz 3 维
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

        # share obs: agents(8 each: pos/type/alive/vel) + stats(6) + tasks(5 each)
        self.agent_share_feat_dim = 8
        self.share_obs_dim = (
                self.max_agents_limit * self.agent_share_feat_dim
                + self.stats_feat_dim
                + (self.max_targets_limit * 5)
        )

        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        self.total_reward = 0.0
        self.cooperation_count = 0.0
        self.completed_tasks = 0.0

    def reset(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """重置环境，必须返回 3 个值"""
        self.current_step = 0
        self.total_reward = 0.0
        self.occupied_positions = set()
        self.cooperation_count = 0.0
        self.completed_tasks = 0.0
        self.episode_pks = []
        self.episode_time_effs = []
        self.episode_paper_effs = []
        self.detected_targets_history = set()

        self.step_buffer_loads = []

        self.type_episode_rewards = {
            NodeType.SENSOR: 0.0,
            NodeType.DECISION: 0.0,
            NodeType.INTERCEPTOR: 0.0
        }

        self.nodes = []
        self.tasks = []
        self.interaction_history = []
        self.message_queue = []
        self.msg_counter = 0
        self.prev_completion_ratio = 0.0

        # ===== 训练/评估模式下的目标数量控制 =====
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

        self._initialize_nodes()

        self.max_mission_count = self.curr_n_target
        self.spawned_tasks_count = 0
        self.completed_tasks = 0.0

        self.tasks = []
        for i in range(self.curr_n_target):
            target_uid = self.n_agents + i
            self._add_node(target_uid, NodeType.TARGET, (0.0, 0.0, 0.0, 0.0), "Target", i)
            task = TaskEntity(
                task_id=f"Task-0-{i}",
                target_node_id=target_uid,
                time_window=self.cfg.task_duration,
                total_time=self.cfg.task_duration,
                start_time=0.0
            )
            task.stage = TaskStage.PENDING
            self.tasks.append(task)

        # 目标节点添加完成后再更新列表。原位置在 _initialize_nodes() 后，会导致列表为空。
        self.target_nodes_list = [n for n in self.nodes if n.node_type == NodeType.TARGET]

        obs = self.get_obs()
        share_obs = self.get_share_obs(obs)
        available_actions = self.get_available_actions()

        return obs, share_obs, available_actions

    def _initialize_nodes(self):
        node_id = 0
        for group in self.cfg.S_GROUPS:
            for i in range(group['count']):
                self._add_node(node_id, NodeType.SENSOR, group['cap'], group['name'], i)
                node_id += 1
        for group in self.cfg.D_GROUPS:
            for i in range(group['count']):
                self._add_node(node_id, NodeType.DECISION, group['cap'], group['name'], i)
                node_id += 1
        for group in self.cfg.I_GROUPS:
            for i in range(group['count']):
                self._add_node(node_id, NodeType.INTERCEPTOR, group['cap'], group['name'], i)
                node_id += 1

    def _add_node(self, uid: int, ntype: NodeType, cap_base: Tuple[float, float, float, float],
                  group_name: str = "", local_id: int = 0):
        """
        连续空间初始化。
        - S/D/I: 在中心 60% 区域随机分布。
        - Target: 在外部区域随机分布。
        - 不再使用网格占用；改用连续距离 min_separation_km 做最小安全间隔检查。
        """
        max_attempts = 1000

        sx = self.cfg.space_size_x
        sy = self.cfg.space_size_y
        sz = self.cfg.space_size_z

        margin_rate = 0.2
        x_min_in = margin_rate * sx
        x_max_in = (1.0 - margin_rate) * sx
        y_min_in = margin_rate * sy
        y_max_in = (1.0 - margin_rate) * sy

        for _ in range(max_attempts):
            if ntype in [NodeType.SENSOR, NodeType.DECISION, NodeType.INTERCEPTOR]:
                x = np.random.uniform(x_min_in, x_max_in)
                y = np.random.uniform(y_min_in, y_max_in)

                # 己方节点按飞行器处理，不再固定在 z=0。
                # 这里放在中低空，避免与目标高度完全重叠过多。
                z = np.random.uniform(0.2 * sz, 0.6 * sz)

            elif ntype == NodeType.TARGET:
                while True:
                    x = np.random.uniform(0.0, sx)
                    y = np.random.uniform(0.0, sy)
                    in_center = (x_min_in <= x < x_max_in) and (y_min_in <= y < y_max_in)
                    if not in_center:
                        break

                z = np.random.uniform(0.2 * sz, sz)

            else:
                x = np.random.uniform(0.0, sx)
                y = np.random.uniform(0.0, sy)
                z = np.random.uniform(0.0, sz)

            candidate_pos = np.array([x, y, z], dtype=np.float32)

            if self._is_position_free(candidate_pos, self_uid=uid):
                node = EndogenousNode(
                    node_id=uid,
                    node_type=ntype,
                    pos=candidate_pos,
                    vel=np.zeros(3, dtype=np.float32),
                    energy=self.cfg.init_energy,
                    raw_capability=np.array(cap_base, dtype=np.float32),
                    group_name=group_name,
                    local_id=local_id
                )

                self.nodes.append(node)
                return

        print(f"Warning: Failed to spawn node {uid} in continuous space")

    def _is_position_free(self, pos: np.ndarray, self_uid: int = -1) -> bool:
        """
        连续空间下的最小安全间隔检查。
        如果与已有存活节点距离小于 min_separation_km，则认为该位置不可用。
        """
        min_sep = getattr(self.cfg, "min_separation_km", 0.05)
        for node in self.nodes:
            if not node.alive:
                continue
            if node.node_id == self_uid:
                continue
            if np.linalg.norm(pos - node.pos) < min_sep:
                return False
        return True

    def _get_node_speed_km_per_step(self, node: EndogenousNode) -> float:
        """
        获取节点固定飞行距离，单位 km/step。
        配置速度单位是 m/s，环境位置单位是 km。
        """
        dt = getattr(self.cfg, "decision_dt", 1.0)

        if node.node_type == NodeType.SENSOR:
            speed_mps = getattr(self.cfg, "speed_S_mps", 50.0)
        elif node.node_type == NodeType.DECISION:
            speed_mps = getattr(self.cfg, "speed_D_mps", 50.0)
        elif node.node_type == NodeType.INTERCEPTOR:
            speed_mps = getattr(self.cfg, "speed_I_mps", 50.0)
        else:
            speed_mps = 0.0

        return float(speed_mps) * float(dt) / 1000.0

    def _limit_turn_by_radius(self, current_vel: np.ndarray, desired_vel: np.ndarray,
                              speed_km_per_step: float) -> np.ndarray:
        """
        根据最大转弯半径限制速度方向变化。

        近似关系：
            Δθ_max = s / R
        其中 s = v * Δt，即当前 step 内飞行距离；R 为转弯半径。
        环境中 speed_km_per_step 已经是 s，radius_km = max_turn_radius_m / 1000。
        """
        eps = 1e-8
        if speed_km_per_step <= eps:
            return np.zeros(3, dtype=np.float32)

        desired_norm = np.linalg.norm(desired_vel)
        if desired_norm <= eps:
            if np.linalg.norm(current_vel) > eps:
                # 没有有效新指令时，保持当前航向继续飞行。
                current_dir = current_vel / np.linalg.norm(current_vel)
                return current_dir.astype(np.float32) * speed_km_per_step
            return np.array([speed_km_per_step, 0.0, 0.0], dtype=np.float32)

        desired_dir = desired_vel / desired_norm
        current_norm = np.linalg.norm(current_vel)

        if current_norm <= eps:
            return desired_dir.astype(np.float32) * speed_km_per_step

        current_dir = current_vel / current_norm
        dot_val = float(np.clip(np.dot(current_dir, desired_dir), -1.0, 1.0))
        theta = math.acos(dot_val)

        radius_km = getattr(self.cfg, "max_turn_radius_m", 1000.0) / 1000.0
        max_theta = speed_km_per_step / max(radius_km, eps)

        if theta <= max_theta:
            return desired_dir.astype(np.float32) * speed_km_per_step

        sin_theta = math.sin(theta)
        if abs(sin_theta) <= eps:
            return current_dir.astype(np.float32) * speed_km_per_step

        new_dir = (
                math.sin(theta - max_theta) / sin_theta * current_dir
                + math.sin(max_theta) / sin_theta * desired_dir
        )
        new_dir = new_dir / max(np.linalg.norm(new_dir), eps)
        return new_dir.astype(np.float32) * speed_km_per_step

    def step(self, actions: List[np.ndarray]):
        """执行一步环境交互，必须返回 6 个值"""
        self.current_step += 1
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        dones = np.zeros(self.n_agents, dtype=bool)

        old_positions = [n.pos.copy() for n in self.nodes[:self.n_agents]]

        move_rewards = self._process_movement(actions)
        rewards += move_rewards

        current_loads = []
        for agent in self.nodes[:self.n_agents]:
            if agent.alive:
                cap = 1
                if agent.node_type == NodeType.SENSOR:
                    cap = self.cfg.max_channels_S
                elif agent.node_type == NodeType.DECISION:
                    cap = self.cfg.max_channels_D
                elif agent.node_type == NodeType.INTERCEPTOR:
                    cap = self.cfg.max_channels_I

                agent.load = agent.get_load(cap)
                current_loads.append(agent.load)

        if current_loads:
            self.step_buffer_loads.append(np.mean(current_loads))

        for i, agent in enumerate(self.nodes[:self.n_agents]):
            if not agent.alive:
                continue

            old_pos = old_positions[i] if i < len(old_positions) else agent.pos

            if agent.node_type == NodeType.INTERCEPTOR:
                target_id = -1
                if len(agent.data_buffer) > 0:
                    msg = agent.data_buffer[0]
                    if isinstance(msg, Message) and 'target_id' in msg.content:
                        target_id = msg.content['target_id']

                if target_id != -1:
                    target = next((n for n in self.nodes if n.node_id == target_id), None)
                    if target is not None and target.alive:
                        curr_dist = np.linalg.norm(agent.pos - target.pos)
                        old_dist = np.linalg.norm(old_pos - target.pos)

                        delta = old_dist - curr_dist
                        rewards[i] += self.cfg.shaping_factor * delta

                        _, _, strike_range = agent.get_actual_ranges(self.cfg)
                        if curr_dist <= strike_range:
                            rewards[i] += 0.05

            elif agent.node_type == NodeType.SENSOR:
                closest_dist = float('inf')
                closest_target = None
                det_range, _, _ = agent.get_actual_ranges(self.cfg)

                for t_node in self.target_nodes_list:
                    if t_node.alive:
                        d = np.linalg.norm(agent.pos - t_node.pos)
                        if d < closest_dist:
                            closest_dist = d
                            closest_target = t_node

                if closest_target is not None and closest_dist < det_range * 2.0:
                    old_dist = np.linalg.norm(old_pos - closest_target.pos)
                    delta = old_dist - closest_dist

                    if closest_dist > det_range * 0.5:
                        rewards[i] += (self.cfg.shaping_factor * 0.5) * delta

        # 基础能耗只对智能体生效
        for i, node in enumerate(self.nodes[:self.n_agents]):
            if node.alive:
                node.consume_energy(self.cfg.energy_cost_hover)
                if not node.alive:
                    rewards[i] -= 1.0

        self._update_tasks()
        self.interaction_history = []
        self._process_actions(actions, rewards)
        self._process_arrived_messages(rewards)

        step_rewards = self._execute_kill_chain()
        rewards += step_rewards

        # Pending penalty 改为比例型
        pending_tasks_count = len([t for t in self.tasks if t.stage == TaskStage.PENDING])

        use_ratio_pending_penalty = getattr(self.cfg, "use_ratio_pending_penalty", True)
        pending_penalty_scale = getattr(
            self.cfg,
            "pending_penalty_scale",
            getattr(self.cfg, "pending_penalty_per_task", 0.0)
        )

        if pending_tasks_count > 0:
            if use_ratio_pending_penalty:
                pending_ratio = pending_tasks_count / max(1.0, float(self.curr_n_target))
                total_penalty = pending_penalty_scale * pending_ratio
            else:
                total_penalty = pending_tasks_count * getattr(self.cfg, "pending_penalty_per_task", 0.0)

            for i in range(self.n_agents):
                if self.nodes[i].alive:
                    rewards[i] -= total_penalty

        # Progress reward
        use_progress_reward = getattr(self.cfg, "use_progress_reward", True)
        progress_reward_scale = getattr(self.cfg, "progress_reward_scale", 20.0)

        completion_ratio = self.completed_tasks / max(1.0, float(self.curr_n_target))
        delta_progress = completion_ratio - getattr(self, "prev_completion_ratio", 0.0)

        if use_progress_reward and delta_progress > 0:
            progress_bonus = progress_reward_scale * delta_progress
            for i in range(self.n_agents):
                if self.nodes[i].alive:
                    rewards[i] += progress_bonus

        self.prev_completion_ratio = completion_ratio

        for i in range(self.n_agents):
            agent_type = self.nodes[i].node_type
            self.type_episode_rewards[agent_type] += rewards[i]

        self.total_reward += float(np.sum(rewards))

        obs = self.get_obs()
        share_obs = self.get_share_obs(obs)
        available_actions = self.get_available_actions()

        is_time_limit = (self.current_step >= self.max_steps)
        is_mission_cleared = (self.completed_tasks >= self.max_mission_count)

        if is_time_limit or is_mission_cleared:
            dones = np.array([True] * self.n_agents)

            total_hops = 0
            chain_count = 0
            for t in self.tasks:
                if t.stage == TaskStage.COMPLETED and t.active_chain_log:
                    hops_sd = max(0, len(t.active_chain_log.get('trace_s_d', [])) - 1)
                    hops_di = max(0, len(t.active_chain_log.get('trace_d_i', [])) - 1)
                    total_hops += (hops_sd + hops_di)
                    chain_count += 1
            avg_hops = total_hops / max(1, chain_count)

            ttc_list = []
            for t in self.tasks:
                if t.first_detection_time is not None:
                    start_t = t.first_detection_time
                    if t.stage == TaskStage.COMPLETED:
                        end_t = t.completion_time
                    elif t.stage == TaskStage.FAILED:
                        end_t = t.completion_time if hasattr(t,
                                                             'completion_time') and t.completion_time else self.current_step
                    else:
                        end_t = self.current_step
                    duration = max(0.0, end_t - start_t)
                    ttc_list.append(duration)
            avg_ttc = np.mean(ttc_list) if ttc_list else 0.0

            avg_load = np.mean(self.step_buffer_loads) if self.step_buffer_loads else 0.0

            total_energy_cost = sum([self.cfg.init_energy - n.energy for n in self.nodes[:self.n_agents]])
            valid_kills = max(1.0, float(self.cooperation_count))
            energy_per_kill = total_energy_cost / valid_kills
            msg_throughput = self.cooperation_count / max(1.0, float(self.msg_counter))

            avg_pk = np.mean(self.episode_pks) if self.episode_pks else 0.0
            avg_time_eff = np.mean(self.episode_time_effs) if self.episode_time_effs else 0.0
            avg_paper_eff = np.mean(self.episode_paper_effs) if self.episode_paper_effs else 0.0

            success_rate = (self.cooperation_count / self.completed_tasks) if self.completed_tasks > 0 else 0.0
            clear_rate = 1.0 if is_mission_cleared else 0.0
            reward_per_target = self.total_reward / max(1.0, float(self.curr_n_target))

            info = {
                "kill_chain/total_reward": self.total_reward,
                "kill_chain/cooperation_count": self.cooperation_count,
                "kill_chain/completed_tasks": self.completed_tasks,
                "kill_chain/success_rate": success_rate,
                "kill_chain/is_success": clear_rate,
                "kill_chain/avg_pk": avg_pk,
                "kill_chain/avg_time_eff": avg_time_eff,
                "kill_chain/paper_temporal_efficiency": avg_paper_eff,

                "kill_chain/current_n_target": float(self.curr_n_target),
                "kill_chain/completion_ratio": completion_ratio,
                "kill_chain/clear_rate": clear_rate,
                "kill_chain/reward_per_target": reward_per_target,

                "kill_chain/reward_S": self.type_episode_rewards[NodeType.SENSOR] / max(1, self.cfg.n_sensor),
                "kill_chain/reward_D": self.type_episode_rewards[NodeType.DECISION] / max(1, self.cfg.n_decision),
                "kill_chain/reward_I": self.type_episode_rewards[NodeType.INTERCEPTOR] / max(1, self.cfg.n_interceptor),

                "metrics/avg_hops": avg_hops,
                "metrics/avg_ttc": avg_ttc,
                "metrics/energy_per_kill": energy_per_kill,
                "metrics/avg_buffer_load": avg_load,
                "metrics/msg_throughput": msg_throughput
            }

            infos = [info.copy() for _ in range(self.n_agents)]
        else:
            dones = np.array([False] * self.n_agents)
            infos = [{} for _ in range(self.n_agents)]

        self._clean_completed_tasks_from_buffers()

        return obs, share_obs, rewards, dones, infos, available_actions

    def _process_movement(self, actions):
        """
        连续三维速度动作处理。

        新动作格式：
            action = [strategy, candidate_idx, vx, vy, vz]

        vx/vy/vz 是 actor 直接输出的连续值。环境执行时：
        1. 先裁剪到 [-velocity_action_clip, velocity_action_clip]；
        2. 将三维向量作为期望速度方向；
        3. 按节点类型固定速度模长，例如当前 S/D/I 均为 50 m/s；
        4. 使用最大转弯半径限制本 step 的航向变化；
        5. 用连续坐标更新位置，不再使用网格。
        """
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        agent_indices = np.arange(self.n_agents)
        np.random.shuffle(agent_indices)

        map_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        map_max = np.array([
            self.cfg.space_size_x,
            self.cfg.space_size_y,
            self.cfg.space_size_z
        ], dtype=np.float32)

        action_clip = float(getattr(self.cfg, "velocity_action_clip", 1.0))

        for i in agent_indices:
            agent = self.nodes[i]
            if not agent.alive:
                continue

            # 读取连续速度方向指令。
            # 如果外部传入动作维度不足，则兼容为保持原航向。
            if len(actions[i]) >= 5:
                desired_cmd = np.asarray(actions[i][2:5], dtype=np.float32)
            else:
                desired_cmd = np.zeros(3, dtype=np.float32)

            desired_cmd = np.clip(desired_cmd, -action_clip, action_clip)

            speed_km_per_step = self._get_node_speed_km_per_step(agent)

            # 固定速度模型：actor 输出的是期望方向，环境把速度模长固定为 speed_km_per_step。
            cmd_norm = np.linalg.norm(desired_cmd)
            if cmd_norm > 1e-8:
                desired_dir = desired_cmd / cmd_norm
                desired_vel = desired_dir * speed_km_per_step
            else:
                # 高斯输出极少严格为 0；这里用于数值保护。
                # 没有有效方向时保持当前航向；若当前也无航向，则默认沿 +X 建立初始航向。
                if np.linalg.norm(agent.vel) > 1e-8:
                    desired_vel = agent.vel / np.linalg.norm(agent.vel) * speed_km_per_step
                else:
                    desired_vel = np.array([speed_km_per_step, 0.0, 0.0], dtype=np.float32)

            actual_vel = self._limit_turn_by_radius(
                current_vel=agent.vel,
                desired_vel=desired_vel,
                speed_km_per_step=speed_km_per_step
            )

            old_pos = agent.pos.copy()
            new_pos = old_pos + actual_vel

            # 边界检查。连续空间没有“网格撞墙”，越界时裁剪到边界并给惩罚。
            out_of_boundary = np.any(new_pos < map_min) or np.any(new_pos > map_max)
            if out_of_boundary:
                new_pos = np.clip(new_pos, map_min, map_max)
                rewards[i] -= 2.0
                agent.consume_energy(self.cfg.energy_cost_collision)
            else:
                agent.consume_energy(self.cfg.energy_cost_move)

            # 连续空间最小安全间隔检查。
            if not self._is_position_free(new_pos, self_uid=agent.node_id):
                new_pos = old_pos
                actual_vel = np.zeros(3, dtype=np.float32)
                rewards[i] -= 2.0
                agent.consume_energy(self.cfg.energy_cost_collision)

            agent.update_position(new_pos)
            agent.update_velocity(actual_vel)

        return rewards

    def _clean_completed_tasks_from_buffers(self):
        """
        [Buffer 持久化管理核心]
        只有当任务状态变为 COMPLETED 时，才从所有节点的 Buffer 中移除该任务相关信息。
        """
        # 1. 收集已完成的任务 ID 和 对应的目标 ID
        completed_task_ids = set()
        completed_target_ids = set()
        for t in self.tasks:
            if t.stage in [TaskStage.COMPLETED, TaskStage.FAILED]:
                completed_task_ids.add(t.task_id)
                completed_target_ids.add(t.target_node_id)

        if not completed_task_ids:
            return

        # 2. 遍历所有节点清理
        for node in self.nodes:
            if not node.alive: continue

            for i in range(len(node.tx_buffer) - 1, -1, -1):
                msg = node.tx_buffer[i]
                # 检查消息内容是否有 task_id 且已过期
                if 'task_id' in msg.content and msg.content['task_id'] in completed_task_ids:
                    node.tx_buffer.pop(i)

            # S 节点 Buffer 存的是 target_id (int)
            if node.node_type == NodeType.SENSOR:
                # 倒序遍历安全删除
                for i in range(len(node.data_buffer) - 1, -1, -1):
                    tid = node.data_buffer[i]
                    if tid in completed_target_ids:
                        node.data_buffer.pop(i)

            # D 节点 Buffer 存的是 dict, 包含 task_id
            elif node.node_type == NodeType.DECISION:
                for i in range(len(node.data_buffer) - 1, -1, -1):
                    item = node.data_buffer[i]
                    if item['task_id'] in completed_task_ids:
                        node.data_buffer.pop(i)

            # I 节点 Buffer 存的是 Message, content含 task_id
            elif node.node_type == NodeType.INTERCEPTOR:
                for i in range(len(node.data_buffer) - 1, -1, -1):
                    msg = node.data_buffer[i]
                    if msg.content['task_id'] in completed_task_ids:
                        node.data_buffer.pop(i)
        for node in self.nodes:
            if node.node_id in completed_target_ids:
                node.alive = False
        pass
        # 3. 仅清理任务列表 (self.tasks 可以删，因为 get_share_obs 是遍历 tasks)
        # self.tasks = [t for t in self.tasks if t.stage not in [TaskStage.COMPLETED, TaskStage.FAILED]]

    def _update_tasks(self):
        """更新所有任务状态"""
        # active_tasks = [t for t in self.tasks if t.stage != TaskStage.COMPLETED]
        active_running_tasks = [
            t for t in self.tasks
            if t.stage not in [TaskStage.COMPLETED, TaskStage.FAILED, TaskStage.PENDING]
        ]
        for task in active_running_tasks:
            # 推进任务时间
            # 这一步通常会减少 task.time_window (剩余时间)
            task.update_time(dt=1.0)

    def _process_actions(self, actions: List[np.ndarray], rewards: np.ndarray):
        """
        处理所有智能体动作
        当前动作格式: [strategy, candidate_idx, vx, vy, vz]
        其中 vx/vy/vz 已在 _process_movement 中处理；本函数只使用 strategy 和 candidate_idx。
        candidate_idx 的语义和 get_available_actions 严格对齐：
        - strategy==1 且 SENSOR: candidate_idx 表示“可探测目标列表中的第 k 个”
        - strategy==2: candidate_idx 表示“可通信邻居列表中的第 k 个”
        - D/I 的 strategy==1 仍不使用 candidate_idx，但改成从 buffer 中挑“更优项”，不再简单 pop(0)
        """
        current_channel_usage = {i: 0 for i in range(self.n_agents)}

        for agent_id, action in enumerate(actions):
            agent = self.nodes[agent_id]

            if not agent.alive:
                continue

            strategy = int(action[0])
            candidate_idx = int(action[1])

            a_det, a_comm, a_str = agent.get_actual_ranges(self.cfg)

            if strategy == 0:
                continue

            # ===============================================================
            # Strategy 1: Produce / Execute
            # ===============================================================
            if strategy == 1:
                # ---------------- SENSOR ----------------
                if agent.node_type == NodeType.SENSOR:
                    # 与 get_available_actions 完全一致：只在“可探测目标集合”里建槽位
                    detectable_targets = []
                    for node in self.nodes:
                        if node.node_type == NodeType.TARGET and node.alive:
                            dist = np.linalg.norm(agent.pos - node.pos)
                            if dist <= a_det:
                                detectable_targets.append((dist, node))
                    detectable_targets.sort(key=lambda x: x[0])

                    if candidate_idx < len(detectable_targets):
                        target_dist, target_node = detectable_targets[candidate_idx]

                        if target_node.node_id not in self.detected_targets_history:
                            rewards[agent_id] += self.cfg.detection_reward
                            self.detected_targets_history.add(target_node.node_id)
                        else:
                            rewards[agent_id] += 0.05

                        interaction = Interaction(
                            agent_id, target_node.node_id,
                            NodeType.SENSOR, NodeType.TARGET,
                            target_dist, "detect", None
                        )
                        self.interaction_history.append(interaction)

                        agent.consume_energy(self.cfg.energy_cost_detection)

                        task_id = None
                        target_task = None
                        for t in self.tasks:
                            if t.target_node_id == target_node.node_id and t.stage != TaskStage.COMPLETED:
                                task_id = t.task_id
                                target_task = t
                                break

                        if target_task is not None and target_task.stage == TaskStage.PENDING:
                            target_task.stage = TaskStage.DETECTED
                            target_task.first_detection_time = float(self.current_step)

                        if len(agent.tx_buffer) < 20 and task_id is not None:
                            q1 = self._calc_q1_detect_prob(agent, target_node)

                            has_duplicate = False
                            for pending_msg in agent.tx_buffer:
                                if (pending_msg.msg_type == "detection_report" and
                                        pending_msg.content['task_id'] == task_id):
                                    has_duplicate = True
                                    break

                            if not has_duplicate:
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
                                        "s_pos": agent.pos,
                                        "t_pos": target_node.pos,
                                        "s_reporter_id": agent_id,
                                        "target_id": target_node.node_id
                                    },
                                    send_time=0,
                                    arrival_time=0,
                                    path_prob=1.0
                                )
                                agent.tx_buffer.append(msg)
                                self.msg_counter += 1

                # ---------------- DECISION ----------------
                elif agent.node_type == NodeType.DECISION:
                    if len(agent.data_buffer) > 0:
                        # 不再简单 pop(0)，而是挑“质量更高 + 更紧急”的情报
                        best_idx = -1
                        best_score = -1.0

                        for idx, intel_data in enumerate(agent.data_buffer):
                            task_id = intel_data['task_id']
                            task = next((t for t in self.tasks if t.task_id == task_id), None)
                            if task is None or task.stage == TaskStage.COMPLETED:
                                continue

                            quality = intel_data.get('confidence', 0.0) * intel_data.get('accumulated_comm_prob', 1.0)
                            urgency = 1.0 - (task.time_window / max(1.0, self.cfg.task_duration))
                            score = quality + 0.2 * urgency

                            if score > best_score:
                                best_score = score
                                best_idx = idx

                        if best_idx != -1:
                            intel_data = agent.data_buffer.pop(best_idx)

                            task_id = intel_data['task_id']
                            task = next((t for t in self.tasks if t.task_id == task_id), None)

                            if task and task.stage != TaskStage.COMPLETED:
                                has_cmd_in_buffer = False
                                for pending_msg in agent.tx_buffer:
                                    if (pending_msg.msg_type == "strike_command" and
                                            pending_msg.content['task_id'] == task_id):
                                        has_cmd_in_buffer = True
                                        break

                                if not has_cmd_in_buffer:
                                    task.assigned_d = agent_id

                                    if task.stage not in [TaskStage.COMPLETED, TaskStage.FAILED]:
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
                                            "origin_confidence": intel_data['confidence'],
                                            "accumulated_comm_prob_s_d": intel_data.get('accumulated_comm_prob', 1.0),
                                            "s_d_trace": intel_data.get('s_d_trace', []),
                                            "q3_prob": q3,
                                            "s_reporter_id": intel_data.get('s_reporter_id', -1)
                                        },
                                        send_time=0,
                                        arrival_time=0,
                                        path_prob=1.0
                                    )

                                    if len(agent.tx_buffer) < 20:
                                        agent.tx_buffer.append(msg)
                                        self.msg_counter += 1
                                        rewards[agent_id] += self.cfg.assignment_reward
                                        agent.consume_energy(self.cfg.energy_cost_comm)

                # ---------------- INTERCEPTOR ----------------
                elif agent.node_type == NodeType.INTERCEPTOR:
                    if len(agent.data_buffer) > 0:
                        # 从指令中挑“链路质量更高 + 目标更合适”的那条
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
                                    msg.content.get("accumulated_comm_prob_s_d", 1.0) *
                                    msg.content.get("q3_prob", 1.0) *
                                    msg.path_prob
                            )

                            dist = np.linalg.norm(agent.pos - target_node.pos)
                            dist_bonus = 1.0 / (1.0 + dist)

                            score = chain_quality + 0.2 * dist_bonus

                            if score > best_score:
                                best_score = score
                                best_idx = idx

                        if best_idx != -1:
                            selected_msg = agent.data_buffer.pop(best_idx)

                            rewards[agent_id] += self.cfg.strike_initiation_reward
                            is_final = self._perform_strike_logic(agent, selected_msg)

                            if not is_final:
                                agent.data_buffer.insert(0, selected_msg)

                            agent.consume_energy(self.cfg.energy_cost_strike)

            # ===============================================================
            # Strategy 2: Comm / Relay
            # ===============================================================
            elif strategy == 2:
                if len(agent.tx_buffer) > 0:
                    # 与 get_available_actions 完全一致：只在“可通信邻居集合”里建槽位
                    neighbor_candidates = []
                    for other in self.nodes:
                        if other.node_id == agent.node_id:
                            continue
                        d = np.linalg.norm(agent.pos - other.pos)
                        link_prob = self._calc_q2_comm_prob(agent, other) if other.alive else 0.0
                        if other.alive and d <= a_comm and link_prob > 0.25:
                            neighbor_candidates.append((d, other, link_prob))
                    neighbor_candidates.sort(key=lambda x: x[0])

                    if candidate_idx < len(neighbor_candidates) and candidate_idx < self.max_candidates:
                        target_dist, target_node, link_prob = neighbor_candidates[candidate_idx]

                        msg = agent.tx_buffer.pop(0)

                        if target_node.node_id in msg.route_trace:
                            agent.consume_energy(self.cfg.energy_cost_comm)
                        else:
                            delay = self._calculate_comm_delay(agent, target_node, target_dist)

                            msg.from_id = agent.node_id
                            msg.to_id = target_node.node_id
                            msg.path_prob *= link_prob
                            msg.hop_count += 1
                            msg.route_trace.append(agent.node_id)

                            msg.send_time = self.current_step
                            msg.arrival_time = self.current_step + delay

                            self.message_queue.append(msg)

                            agent.consume_energy(self.cfg.energy_cost_comm)

                            interaction = Interaction(
                                agent_id, target_node.node_id,
                                agent.node_type, target_node.node_type,
                                target_dist, "comm", None
                            )
                            self.interaction_history.append(interaction)
                    else:
                        # 选到空槽位，不执行
                        pass

    def _determine_interaction_type(self, from_type: NodeType, to_type: NodeType) -> str:
        """确定交互类型"""
        if from_type == NodeType.SENSOR and to_type == NodeType.TARGET:
            return "detect"
        elif from_type == NodeType.SENSOR and to_type == NodeType.DECISION:
            return "report"
        elif from_type == NodeType.DECISION and to_type == NodeType.INTERCEPTOR:
            return "command"
        elif from_type == NodeType.INTERCEPTOR and to_type == NodeType.TARGET:
            return "strike"
        else:
            return "comm"

    # ========== 通信时延模型 ==========

    def _calculate_transmission_delay(self, sender: EndogenousNode, receiver: EndogenousNode, distance: float) -> float:
        """
        计算传输时延

        传输时延 = 距离 / 有效数据速率
        有效数据速率 = base_rate × min(发送方通信能力, 接收方通信能力) / 10

        Args:
            sender: 发送节点
            receiver: 接收节点
            distance: 距离(km)

        Returns:
            传输时延(步数)
        """
        # 获取双方有效通信能力（索引1）
        sender_comm_cap = sender.get_effective_capability()[1]
        receiver_comm_cap = receiver.get_effective_capability()[1]

        # 木桶效应
        effective_cap = min(sender_comm_cap, receiver_comm_cap)

        # 归一化到0-1
        # 归一化能力因子 (假设标准能力值为 10.0)
        cap_factor = effective_cap / 10.0

        # 计算实际物理传输速率 (km/step)
        # base_data_rate 在 config.py 中定义
        # 意味着：能力越强，cap_factor 越大，effective_rate 越快
        effective_rate = self.cfg.base_data_rate * cap_factor

        # 异常处理：如果由于干扰或损坏导致能力归零，则无法通信
        if effective_rate <= 0:
            return 999.0  # 无法通信

        # 传输时延
        trans_delay = distance / effective_rate

        # 距离越远，时延越大；速率越高，时延越小
        return int(np.ceil(trans_delay))

    def _calculate_queuing_delay(self, receiver: EndogenousNode) -> float:
        """
        计算队列延迟

        队列延迟 = f(接收方负载)
        负载越高，消息需要排队等待处理

        Args:
            receiver: 接收节点

        Returns:
            队列延迟(步数)
        """
        # load = 当前占用通道数 / 最大通道数
        load = receiver.load

        # 1. 过载保护 / 拒绝服务 (DoS)
        if load >= 0.99:
            return 999.0  # 拒绝服务

        # 2. 指数增长模型 (M/M/1 排队近似)
        # 公式: Delay = k * (load / (1 - load))
        # 系数 2.0 是一个调节因子，控制延迟增长的陡峭程度。
        queue_delay = 2.0 * load / (1.0 - load)

        return int(np.ceil(queue_delay))

    import numpy as np

    def calculate_recon_time(self,sensor_pos, target_pos):
        """
        计算侦察时间 (方案一：基于距离)

        Args:
            sensor_pos (np.array): S节点坐标 [x, y, z]
            target_pos (np.array): 目标坐标 [x, y, z]
            noise_level (float): 噪声比例，默认 10%

        Returns:
            float: 侦察耗时 (秒)
        """
        # 1. 计算欧几里得距离 (假设坐标单位是 km)
        distance = np.linalg.norm(sensor_pos - target_pos)

        noise_level = 0.1
        # 2. 设定参数 (根据你的场景尺度调整)
        t_base = 2.0  # 基础处理时间：2秒
        mu = 0.05  # 距离系数：每100km增加5秒 (即 0.05 s/km)

        # 3. 计算理论时间
        t_theoretical = t_base + (mu * distance)

        # 4. 加入随机噪声 (模拟战场不确定性)
        # 生成一个 [1-noise, 1+noise] 的随机系数
        random_factor = np.random.uniform(1 - noise_level, 1 + noise_level)

        t_final = t_theoretical * random_factor

        return int(np.ceil(t_final))

    def _calculate_comm_delay(self, sender: EndogenousNode, receiver: EndogenousNode, distance: float) -> int:
        """
        计算总通信时延

        总时延 = 传输时延 + 队列延迟

        Returns:
            总时延(步数, 整数)
        """
        trans_delay = self._calculate_transmission_delay(sender, receiver, distance)
        queue_delay = self._calculate_queuing_delay(receiver)

        total_delay = trans_delay + queue_delay

        # 至少1步
        return max(1, int(np.ceil(total_delay)))

    # ========== 决策时延模型 ==========

    def _calculate_decision_delay(self, d_node: EndogenousNode, task_complexity: int) -> int:
        """
        计算决策处理时延
        决策时延 = f(有效决策能力, 任务复杂度, 负载)
        注意：工作模式不直接影响时延，而是通过影响有效决策能力来间接影响
        Args:
            d_node: 决策节点
            task_complexity: 任务复杂度（需要协调的节点数量）

        Returns:
            决策时延(步数)
        """
        # 1. 基础时间常数
        base_time = self.cfg.base_decision_time

        # 2. 能力因子 (Capability Factor) - 核心瓶颈
        # 这一步体现了 WorkMode 的作用：
        # 如果 d_node 开了 HIGH_PERF 模式，decision_cap 变大，从而显著降低时延
        decision_cap = d_node.get_effective_capability()[2]

        # 反比模型: 能力越强，处理时间越短
        # (10.0 / (x + 1.0)) 使得能力提升带来的收益呈现边际递减效应
        cap_factor = 10.0 / (decision_cap + 1.0)

        # 任务越复杂（例如需要组织多枚导弹进行饱和攻击），需要计算的协同参数越多
        # 公式: 1 + 0.3 * N (假设系数为0.3)
        complexity_factor = 1.0 + self.cfg.complexity_factor_per_node * task_complexity

        # 模拟 CPU/操作员 的繁忙程度。如果节点正在处理大量其他通信或计算，新任务的处理会变慢。
        # load范围 0~1。如果满载 (load=1.0)，时间会变为原来的 3 倍 (1 + 2.0)
        load_factor = 1.0 + d_node.load * 2.0

        # 总时间
        total_time = base_time * cap_factor * complexity_factor * load_factor

        return max(1, int(np.ceil(total_time)))

    # ========== 武器飞行时延模型 ==========

    def _calculate_flight_time(self, interceptor: EndogenousNode, target: EndogenousNode) -> int:
        """
        计算武器飞行时间
        飞行时间 = 距离 / 武器速度
        武器速度 = base_speed + 有效打击能力 × speed_per_cap

        Args:
            interceptor: 拦截器节点
            target: 目标节点

        Returns:
            飞行时间(步数)
        """
        distance = np.linalg.norm(interceptor.pos - target.pos)

        # 获取有效打击能力（受 WorkMode 调节）
        strike_cap = interceptor.get_effective_capability()[3]

        # 计算武器速度: 基础速度 + 能力增益
        # 例如: 500 + 10.0 * 50 = 1000 km/step
        weapon_speed = self.cfg.base_weapon_speed + strike_cap * self.cfg.weapon_speed_per_cap

        if weapon_speed <= 0:
            return 999

        # 飞行时间
        flight_time = distance / weapon_speed

        return max(1, int(np.ceil(flight_time)))

    # ========== 概率计算（保持原有逻辑）==========

    def _calc_q1_detect_prob(self, sensor: EndogenousNode, target: EndogenousNode) -> float:
        """
            [公式 1.22] Q1: 侦察发现概率 (Detection Probability)

            模拟雷达/传感器的物理特性。

            影响因素:
            1. 距离衰减: 遵循四次方衰减律 (Radar Equation近似)，距离越远概率下降极快。
            2. 传感器能力: 只有 Sensor 节点的侦察能力(Index 0)发挥作用。
            """
        dist = np.linalg.norm(sensor.pos - target.pos)
        det_range, _, _ = sensor.get_actual_ranges(self.cfg)
        # cap1 = self.cfg.detection_range

        if dist > det_range or det_range <= 0:
            return 0.0

        # 物理衰减基于动态探测半径
        return np.exp(-0.69 * ((dist / det_range) ** 4)) * (sensor.get_effective_capability()[0] / 10.0)

    def _calc_q2_comm_prob(self, node_i: EndogenousNode, node_j: EndogenousNode) -> float:
        """
        [修正版] 区域干扰逻辑：压制有效通信半径 (Range Suppression)
        """
        dist = np.linalg.norm(node_i.pos - node_j.pos)
        is_jammed = False  # <--- 1. 必须在循环外初始化！
        # 1. 获取原始能力
        _, comm_i, _ = node_i.get_actual_ranges(self.cfg)  # 发送者半径
        _, comm_j, _ = node_j.get_actual_ranges(self.cfg)  # 接收者半径

        # ==================== [修改] 干扰判定 ====================
        # 检查发送方是否被干扰
        for node in self.nodes:
            if node.node_type == NodeType.TARGET and node.alive:
                dist_to_jammer = np.linalg.norm(node_i.pos - node.pos)  # 计算一下距离
                if dist_to_jammer <= self.cfg.jamming_radius:
                    is_jammed = True
                    # [DEBUG] 打印干扰触发信息
                    # print(f"⚠️ Agent {node_i.node_id} 被 Target {node.node_id} 干扰! 距离: {dist_to_jammer:.2f} < {self.cfg.jamming_radius}")
                    break
        # 如果被干扰，有效通信半径直接打骨折 (例如只剩 15%)
        # 这比直接乘概率更符合物理规律 (距离越远衰减越剧烈)
        if is_jammed:
            comm_i *= (1.0 - self.cfg.jamming_intensity)
        # 2. 计算合成半径 (木桶效应)
        eff_range = min(comm_i, comm_j)
        if eff_range <= 0: return 0.0
        # 3. 计算最终概率
        base_prob = (eff_range ** 2) / (dist ** 2 + eff_range ** 2)
        return base_prob

    def _calc_q3_command_prob(self, d_node: EndogenousNode, comm_prob_d_i: float) -> float:
        """
            [公式 1.24] Q3: 指控/决策正确率 (Command Efficiency)

            模拟 C2 节点对情报的理解和指令生成的准确性。

            公式逻辑:
            Q3 = 决策能力 * 前序链路质量
            意味着：即使前面的通信很完美，如果指挥员(D节点)是个"笨蛋"(能力低)，最终指令的有效性也会打折。
            """
        # 获取有效决策能力
        # cap3 = d_node.get_effective_capability()[2] / 10.0
        cap3 = d_node.get_effective_capability()[2]

        # 这里的逻辑是：决策质量不仅取决于自身脑力(cap3)，
        # 假设 cap3 是 0.0~1.0 的系数，或者在外部有归一化。
        q3 = cap3 * comm_prob_d_i

        return float(q3)

    def _calc_q4_strike_prob(self, interceptor: EndogenousNode, target: EndogenousNode) -> float:
        """
        [修正版] 打击概率计算
        优化点：
        1. 能力(Capability)修正 CEP 而不是直接乘概率，体现制导水平差异。
        2. 增加超射程的惩罚因子（动力衰减）。
        """
        dist = np.linalg.norm(interceptor.pos - target.pos)

        # 获取参考射程
        _, _, strike_range = interceptor.get_actual_ranges(self.cfg)
        ref_range = max(1.0, strike_range)

        base_cep = self.cfg.weapon_cep
        lethal_radius = self.cfg.target_radius

        if base_cep <= 0: return 1.0

        # --- 1. 计算能力修正系数 ---
        # 假设能力值 10.0 为标准。
        # 能力 12.0 -> CEP 变小 (0.83倍) -> 精度更高
        # 能力 8.0  -> CEP 变大 (1.25倍) -> 精度更低
        current_cap = interceptor.get_effective_capability()[3]
        cap_factor = current_cap / 10.0
        cap_factor = max(0.1, cap_factor)  # 保护

        # --- 2. 动态 CEP 计算 ---
        # 基础距离发散 (模拟角度误差)
        dist_ratio = dist / ref_range
        # 综合 CEP：距离越远CEP越大，能力越强CEP越小
        dynamic_cep = (base_cep * max(0.1, dist_ratio)) / cap_factor

        # --- 3. 物理杀伤公式 (Carlton Function) ---
        exponent = (lethal_radius / dynamic_cep) ** 2
        q4_physical = 1.0 - (0.5 ** exponent)

        # --- 4. (可选) 动力学惩罚 ---
        # 如果超出射程，不仅仅是精度问题，还有动能不足的问题
        # 超过 1.2 倍射程后，概率指数级下降
        range_penalty = 1.0
        if dist > ref_range * 1.5:
            range_penalty = np.exp(-(dist - ref_range * 1.5) / (ref_range * 0.6))

        final_prob = q4_physical * range_penalty

        return float(final_prob)

    # ========== 杀伤链执行逻辑（带通信时延）==========
    def _execute_kill_chain(self) -> np.ndarray:
        """
                执行杀伤链全流程逻辑（最终修复版 - 分层处理机制）
                逻辑顺序：
                1. 处理消息入库
                2. 发放延迟奖励
                3. 阶段一：感知融合 (处理 Detect) -> 确保 S 节点间完成信息共享
                4. 阶段二：通信决策 (处理 Report/Command) -> 发送融合后的数据
                5. 阶段三：打击执行 (处理 I 节点 Buffer)
                6. 全局惩罚与超时检查
                """
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        # ==========================================
        # 1. 处理到达消息 (存入接收方 Buffer)
        # ==========================================
        # self._process_arrived_messages()

        # ==========================================
        # 2. 处理延迟奖励 (Pending Rewards)
        # ==========================================
        for task in self.tasks:
            if hasattr(task, 'pending_reward_distribution') and task.pending_reward_distribution:
                contributors, share = task.pending_reward_distribution
                for cid in contributors:
                    if cid < self.n_agents: rewards[cid] += share
                task.pending_reward_distribution = None

        # ==========================================
        # 全局惩罚与超时检查
        # ==========================================
        active_tasks = [t for t in self.tasks if t.stage != TaskStage.COMPLETED]

        # # 基础时间流逝惩罚 (促使尽快完成)
        if active_tasks:
            # 让所有活着的人都承担完整的时间压力，迫使它们协作
            step_penalty = self.cfg.w_time
            for i in range(self.n_agents):
                rewards[i] -= step_penalty

        # 任务硬超时检查
        for task in active_tasks:
            if task.time_window <= 0:
                # 打印超时战报 (让您知道它是怎么挂的)
                last_pk = task.final_kill_prob if hasattr(task, 'final_kill_prob') else 0.0
                time_cost = float(self.current_step - task.start_time)
                # self._print_tsdit_chain(task, last_pk, time_cost, False, "窗口耗尽-超时")
                self.completed_tasks += 1
                task.stage = TaskStage.FAILED

                # 失败惩罚
                penalty = self.cfg.timeout_penalty

                penalty_per_agent = penalty / max(1, self.n_agents)  # 或者直接每人扣一个固定值

                for i in range(self.n_agents):
                    if self.nodes[i].alive:
                        rewards[i] += penalty_per_agent

                # contributors = task.assigned_s_list + [task.assigned_d] + task.assigned_i_list
                # contributors = [cid for cid in contributors if cid < self.n_agents]
                #
                # if contributors:
                #     share = penalty / len(contributors)
                #     for cid in contributors:
                #         rewards[cid] += share

        return rewards

    def _perform_strike_logic(self, i_node, msg):
        """
        I 节点执行打击的结算逻辑
        """
        task_id = msg.content["task_id"]
        task = next((t for t in self.tasks if t.task_id == task_id), None)

        if task:
            if i_node.node_id not in task.assigned_i_list:
                task.assigned_i_list.append(i_node.node_id)

            if task.stage != TaskStage.COMPLETED and task.stage != TaskStage.FAILED:
                task.set_stage(TaskStage.STRIKING, self.current_step)

            if i_node.node_id not in task.assigned_i_list:
                task.assigned_i_list.append(i_node.node_id)

            target_node = next((n for n in self.nodes if n.node_id == task.target_node_id), None)

            p1 = msg.content.get("origin_confidence", 0.0)
            p2 = msg.content.get("accumulated_comm_prob_s_d", msg.content.get("origin_comm_s_d", 1.0))
            p3 = msg.content.get("q3_prob", msg.content.get("q3_command_prob", 1.0))
            p4 = self._calc_q4_strike_prob(i_node, target_node)
            p_link_2 = msg.path_prob

            pk = p1 * p2 * p3 * p_link_2 * p4
            self.episode_pks.append(pk)

            trace_s_d = msg.content.get('s_d_trace', [])
            trace_d_i = msg.route_trace + [i_node.node_id]

            task.active_chain_log = {
                's': msg.content['s_reporter_id'],
                'd': task.assigned_d,
                'i': i_node.node_id,
                'probs': [p1, p2, p3, p4],
                'trace_s_d': trace_s_d,
                'trace_d_i': trace_d_i,
                'prob_s_d': p2,
                'prob_d_i': msg.path_prob
            }
            task.final_kill_prob = pk

            flight = self._calculate_flight_time(i_node, target_node)
            arrival_time = self.current_step + flight

            start_t = task.first_detection_time if task.first_detection_time is not None else task.start_time
            time_cost = arrival_time - start_t
            time_cost = max(0.0, time_cost)

            paper_eff = 1.0 / max(1.0, time_cost)
            self.episode_paper_effs.append(paper_eff)

            if task.stage != TaskStage.COMPLETED:
                # 这里改成看全链路 pk，而不是只看 p4
                if pk >= self.cfg.min_kill_probability:
                    task.stage = TaskStage.COMPLETED
                    task.completion_time = arrival_time
                    self.completed_tasks += 1
                    self.cooperation_count += 1

                    contributors = [msg.content['s_reporter_id'], task.assigned_d, i_node.node_id]
                    contributors = [c for c in contributors if c < self.n_agents]
                    if contributors:
                        base = task.accumulated_value + self.cfg.finish_const_reward
                        max_window = self.cfg.task_duration
                        time_eff = max(0.0, (max_window - time_cost) / max_window)
                        self.episode_time_effs.append(time_eff)

                        total = base * pk * (0.7 + 0.3 * time_eff)
                        task.pending_reward_distribution = (contributors, total / len(contributors))
                else:
                    penalty = self.cfg.soft_failure_penalty
                    contributors = [i_node.node_id]
                    contributors = [c for c in contributors if c < self.n_agents]

                    if contributors:
                        share = penalty / len(contributors)
                        task.pending_reward_distribution = (contributors, share)


    def _process_arrived_messages(self, rewards):
        # 筛选到达消息
        arrived = [m for m in self.message_queue if m.has_arrived(self.current_step)]
        self.message_queue = [m for m in self.message_queue if not m.has_arrived(self.current_step)]

        for msg in arrived:
            receiver = next((n for n in self.nodes if n.node_id == msg.to_id), None)
            if not receiver or not receiver.alive:
                return  # 接收者不存在或已阵亡，消息丢失

            # === 判断1: 终于到达目标类型了吗？ (S->...->D 或 D->...->I) ===
            if receiver.node_type == msg.target_node_type:
                # 逻辑: 基础奖励 * 链路质量 - 跳数惩罚
                # quality_weighted_reward = self.cfg.reward_routing_success * msg.path_prob                # 1. 质量加权: 链路概率越高(msg.path_prob)，奖励越高
                quality_weighted_reward = self.cfg.reward_routing_success
                hop_penalty = msg.hop_count * 0.1                                                        # 2. 跳数惩罚: 每多一跳扣除 0.1 (引导寻找最短路径)
                final_success_reward = max(0.0, quality_weighted_reward - hop_penalty)                   # 3. 最终奖励 (保底为 0，不给负分)
                # 1. 奖励发起人 (业务源头：S 或 D)
                origin_id = msg.initial_source_id
                if origin_id < self.n_agents:
                    rewards[origin_id] += final_success_reward * 0.5

                # 2. 奖励最后一跳
                last_hop_id = msg.from_id
                if last_hop_id < self.n_agents:
                    rewards[last_hop_id] += final_success_reward

                # 3. 奖励中间路径 (路由节点)
                routing_reward = self.cfg.reward_routing_success * 0.5
                for hop_node_id in msg.route_trace:
                    if hop_node_id < self.n_agents and hop_node_id != last_hop_id:
                        rewards[hop_node_id] += routing_reward

                # 2. 消息入库 (进入业务层)
                if receiver.node_type == NodeType.DECISION:
                    # D 收到情报，保存 msg.path_prob 作为 Q2
                    # content 里需要把 path_prob 记下来，供后续 Command 使用
                    intel_data = msg.content
                    task_id = intel_data['task_id']
                    intel_data['accumulated_comm_prob'] = msg.path_prob  # 存下累积概率

                    # [新增] 保存 S->...->D 的物理路径
                    # msg.route_trace 是之前的路径，加上 receiver.node_id 就是完整路径
                    full_trace = msg.route_trace + [receiver.node_id]
                    intel_data['s_d_trace'] = full_trace

                    # 计算这份情报的“综合质量”
                    # 质量 = 原始侦察置信度(Q1) × 传输链路累积概率(Path_Prob)
                    current_quality = intel_data['confidence'] * msg.path_prob

                    # 检查 Buffer 中是否已有该任务的情报
                    existing_idx = -1
                    for idx, item in enumerate(receiver.data_buffer):
                        if item['task_id'] == task_id:
                            existing_idx = idx
                            break
                    if existing_idx != -1:
                        # 【已有旧情报】：比较质量
                        old_item = receiver.data_buffer[existing_idx]
                        old_quality = old_item['confidence'] * old_item.get('accumulated_comm_prob', 1.0)
                        if current_quality > old_quality:
                            # 新情报链路更好 -> 替换！
                            # 记录累积概率供后续 D->I 使用
                            intel_data['accumulated_comm_prob'] = msg.path_prob
                            receiver.data_buffer[existing_idx] = intel_data
                            # print(f"D{receiver.node_id} 优化情报: {task_id} (P_total: {old_quality:.2f} -> {current_quality:.2f})")
                            # [新增] 强化追责与奖励：D 节点成功优化了情报
                            if receiver.node_id < self.n_agents:
                                # 给予一个瞬时的“情报融合奖励”，分值为路由成功奖的 20%
                                rewards[receiver.node_id] += self.cfg.reward_routing_success * 0.2
                        else:
                            # 旧情报更好 -> 丢弃新消息
                            pass
                    else:
                        # 【新任务】：直接入库
                        if len(receiver.data_buffer) < self.cfg.max_channels_D:
                            intel_data['accumulated_comm_prob'] = msg.path_prob
                            receiver.data_buffer.append(intel_data)
                elif receiver.node_type == NodeType.INTERCEPTOR:
                    # 1. 提取指令中的关键信息
                    task_id = msg.content['task_id']

                    # 2. 计算当前这条指令的“预期全链路杀伤概率”
                    prob_s_d = msg.content.get('accumulated_comm_prob_s_d', 1.0)
                    prob_q3 = msg.content.get('q3_prob', 1.0)
                    prob_d_i = msg.path_prob

                    current_chain_quality = prob_s_d * prob_q3 * prob_d_i

                    # 3. 检查 Buffer 中是否已有针对该任务的指令
                    existing_idx = -1
                    for idx, item in enumerate(receiver.data_buffer):
                        # I 节点的 buffer 存的是 Message 对象
                        if item.content['task_id'] == task_id:
                            existing_idx = idx
                            break

                    if existing_idx != -1:
                        # === 情况 A: 已有指令，进行“择优” ===
                        old_msg = receiver.data_buffer[existing_idx]

                        # 计算旧指令的质量
                        old_prob_s_d = old_msg.content.get('accumulated_comm_prob_s_d', 1.0)
                        old_prob_q3 = old_msg.content.get('q3_prob', 1.0)
                        old_prob_d_i = old_msg.path_prob

                        old_chain_quality = old_prob_s_d * old_prob_q3 * old_prob_d_i

                        # 如果新指令的链路概率更高，则替换旧指令
                        if current_chain_quality > old_chain_quality:
                            receiver.data_buffer[existing_idx] = msg
                            # (可选) 打印日志观察优化效果
                            # print(f"I{receiver.node_id} 优化指令: {task_id} (P: {old_chain_quality:.3f} -> {current_chain_quality:.3f})")
                        else:
                            # 旧指令更好，丢弃当前这条较差的指令
                            pass

                    else:
                        # === 情况 B: 新任务指令，直接入库 ===
                        if len(receiver.data_buffer) < self.cfg.max_channels_I:
                            receiver.data_buffer.append(msg)

            # === 判断2: 还是中途？需要转发吗？ ===
            else:
                # 接收者不是目标类型 (例如 S 发给了 S，或者 S 发给了 I)
                # 放入接收者的发送缓冲区 (TX Buffer)，等待它下一跳转发

                if msg.hop_count < 10:  # TTL 限制
                    if len(receiver.tx_buffer) < 20:
                        receiver.tx_buffer.append(msg)
                else:
                    pass  # 超时丢弃

    def get_obs(self) -> List[np.ndarray]:
        """
        稳定槽位版本的局部观测：
        [self(13), buffer(7), compass(6), stats(6), tasks(max_targets_limit*7), neighbors(5*7)]
        task feat = [valid, time_left, stage, visible, x, y, z]
        """
        obs_list = []
        K_NEIGHBORS = self.max_candidates
        NEIGHBOR_DIM = 7

        all_pos = np.array([n.pos for n in self.nodes], dtype=np.float32)
        all_alive = np.array([1.0 if n.alive else 0.0 for n in self.nodes], dtype=np.float32)
        all_types = np.array([n.node_type.value / 3.0 for n in self.nodes], dtype=np.float32)
        all_loads = np.array([n.load for n in self.nodes], dtype=np.float32)

        agent_pos = all_pos[:self.n_agents]
        diff_matrix = agent_pos[:, np.newaxis, :] - all_pos[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)

        diff_matrix_norm = (diff_matrix / self.map_size).astype(np.float32)
        dist_matrix_norm = (dist_matrix / max(self.max_span, 1e-6)).astype(np.float32)

        # ===== 全局统计特征 =====
        pending_count = sum(1 for t in self.tasks if t.stage == TaskStage.PENDING)
        detected_count = sum(1 for t in self.tasks if t.stage == TaskStage.DETECTED)
        assigned_count = sum(1 for t in self.tasks if t.stage in [TaskStage.ASSIGNED, TaskStage.STRIKING])
        completed_count = sum(1 for t in self.tasks if t.stage == TaskStage.COMPLETED)
        active_count = sum(1 for t in self.tasks if t.stage not in [TaskStage.COMPLETED, TaskStage.FAILED])

        denom = max(1.0, float(self.curr_n_target))
        stats_feat = np.array([
            self.curr_n_target / max(1.0, float(self.max_targets_limit)),
            active_count / denom,
            pending_count / denom,
            detected_count / denom,
            assigned_count / denom,
            completed_count / denom,
        ], dtype=np.float32)

        # ===== 稳定任务槽位 =====
        task_state_flat = np.zeros(self.max_observed_tasks * self.task_feat_dim, dtype=np.float32)
        sorted_tasks = sorted(self.tasks, key=lambda t: t.target_node_id)

        for task in sorted_tasks:
            slot = task.target_node_id - self.n_agents
            if slot < 0 or slot >= self.max_observed_tasks:
                continue

            # actor 只关心未完成/未失败任务；已完成/失败槽位置零
            if task.stage in [TaskStage.COMPLETED, TaskStage.FAILED]:
                continue

            target_node = next((n for n in self.nodes if n.node_id == task.target_node_id), None)

            is_visible = 0.0
            t_pos_x, t_pos_y, t_pos_z = 0.0, 0.0, 0.0
            if task.stage != TaskStage.PENDING and target_node is not None:
                is_visible = 1.0
                t_pos_x = target_node.pos[0] / self.cfg.space_size_x
                t_pos_y = target_node.pos[1] / self.cfg.space_size_y
                t_pos_z = target_node.pos[2] / self.cfg.space_size_z

            feat = np.array([
                1.0,
                task.time_window / max(1.0, self.cfg.task_duration),
                task.stage.value / 5.0,
                is_visible,
                t_pos_x,
                t_pos_y,
                t_pos_z
            ], dtype=np.float32)

            start_idx = slot * self.task_feat_dim
            end_idx = start_idx + self.task_feat_dim
            task_state_flat[start_idx:end_idx] = feat

        # ===== 为每个智能体组装观测 =====
        for i in range(self.n_agents):
            agent = self.nodes[i]

            has_msg = 1.0 if len(agent.tx_buffer) > 0 else 0.0
            speed_norm = max(self._get_node_speed_km_per_step(agent), 1e-6)
            self_state = np.array([
                agent.energy / max(1.0, self.cfg.init_energy),
                agent.load,
                has_msg,
                agent.raw_capability[0] / 10.0,
                agent.raw_capability[1] / 10.0,
                agent.raw_capability[2] / 10.0,
                agent.raw_capability[3] / 10.0,
                agent.pos[0] / self.cfg.space_size_x,
                agent.pos[1] / self.cfg.space_size_y,
                agent.pos[2] / self.cfg.space_size_z,
                agent.vel[0] / speed_norm,
                agent.vel[1] / speed_norm,
                agent.vel[2] / speed_norm
            ], dtype=np.float32)

            # ===== compass =====
            compass_feat = np.zeros(self.compass_dim, dtype=np.float32)

            # 最近关键队友
            target_mate_type = None
            if agent.node_type == NodeType.DECISION:
                target_mate_type = NodeType.INTERCEPTOR
            elif agent.node_type == NodeType.INTERCEPTOR:
                target_mate_type = NodeType.DECISION

            if target_mate_type is not None:
                best_dist = float('inf')
                best_mate_idx = -1
                for idx, node in enumerate(self.nodes[:self.n_agents]):
                    if idx == i:
                        continue
                    if node.node_type == target_mate_type and node.alive:
                        d = dist_matrix[i, idx]
                        if d < best_dist:
                            best_dist = d
                            best_mate_idx = idx

                if best_mate_idx != -1:
                    compass_feat[0] = diff_matrix_norm[i, best_mate_idx, 0]
                    compass_feat[1] = diff_matrix_norm[i, best_mate_idx, 1]
                    compass_feat[2] = diff_matrix_norm[i, best_mate_idx, 2]

            # 最近已知目标
            best_t_idx = -1
            best_t_dist = float('inf')

            if agent.node_type == NodeType.SENSOR:
                det_range, _, _ = agent.get_actual_ranges(self.cfg)
                for idx, node in enumerate(self.nodes):
                    if node.node_type == NodeType.TARGET and node.alive:
                        d = dist_matrix[i, idx]
                        if d <= det_range and d < best_t_dist:
                            best_t_dist = d
                            best_t_idx = idx

            elif agent.node_type == NodeType.INTERCEPTOR:
                known_target_ids = set()
                for item in list(agent.data_buffer) + list(agent.tx_buffer):
                    if isinstance(item, Message) and 'target_id' in item.content:
                        known_target_ids.add(item.content['target_id'])

                for idx, node in enumerate(self.nodes):
                    if node.node_type == NodeType.TARGET and node.alive and node.node_id in known_target_ids:
                        d = dist_matrix[i, idx]
                        if d < best_t_dist:
                            best_t_dist = d
                            best_t_idx = idx

            if best_t_idx != -1:
                compass_feat[3] = diff_matrix_norm[i, best_t_idx, 0]
                compass_feat[4] = diff_matrix_norm[i, best_t_idx, 1]
                compass_feat[5] = diff_matrix_norm[i, best_t_idx, 2]

            # ===== buffer feat =====
            buffer_feat = np.zeros(self.buffer_feat_dim, dtype=np.float32)
            target_msg = None
            if len(agent.data_buffer) > 0 and isinstance(agent.data_buffer[0], Message):
                target_msg = agent.data_buffer[0]
            elif len(agent.tx_buffer) > 0 and isinstance(agent.tx_buffer[0], Message):
                target_msg = agent.tx_buffer[0]

            if target_msg is not None:
                buffer_feat[0] = target_msg.target_node_type.value / 3.0
                if 'confidence' in target_msg.content:
                    buffer_feat[1] = target_msg.content['confidence']
                elif 'q3_prob' in target_msg.content:
                    buffer_feat[1] = target_msg.content['q3_prob']
                else:
                    buffer_feat[1] = 0.5

                buffer_feat[2] = target_msg.path_prob
                buffer_feat[3] = min(1.0, target_msg.hop_count / 10.0)

                t_id = target_msg.content.get('target_id')
                if t_id is not None:
                    t_node = next((n for n in self.nodes if n.node_id == t_id), None)
                    if t_node is not None:
                        rel_vec = (t_node.pos - agent.pos)
                        buffer_feat[4] = rel_vec[0] / self.cfg.space_size_x
                        buffer_feat[5] = rel_vec[1] / self.cfg.space_size_y
                        buffer_feat[6] = rel_vec[2] / self.cfg.space_size_z

            # ===== neighbor feat =====
            dists = dist_matrix[i].copy()
            dists[i] = np.inf
            sorted_indices = np.argsort(dists)[:K_NEIGHBORS]

            neighbor_feats_list = []
            for idx in sorted_indices:
                if dists[idx] == np.inf:
                    neighbor_feats_list.extend([0.0] * NEIGHBOR_DIM)
                    continue

                n_feat = [
                    all_types[idx],
                    diff_matrix_norm[i, idx, 0],
                    diff_matrix_norm[i, idx, 1],
                    diff_matrix_norm[i, idx, 2],
                    dist_matrix_norm[i, idx],
                    all_alive[idx],
                    all_loads[idx]
                ]
                neighbor_feats_list.extend(n_feat)

            if len(neighbor_feats_list) < K_NEIGHBORS * NEIGHBOR_DIM:
                remains = K_NEIGHBORS * NEIGHBOR_DIM - len(neighbor_feats_list)
                neighbor_feats_list.extend([0.0] * remains)

            neighbor_state = np.array(neighbor_feats_list, dtype=np.float32)

            obs = np.concatenate([
                self_state,
                buffer_feat,
                compass_feat,
                stats_feat,
                task_state_flat,
                neighbor_state
            ])
            obs_list.append(obs)

        return obs_list

    def _spawn_task(self):
        """生成新任务"""
        if self.spawned_tasks_count >= self.max_mission_count:
            return

        # active_tasks = [t for t in self.tasks if t.stage not in [TaskStage.COMPLETED, TaskStage.FAILED]]
        # if len(active_tasks) > 0:
        #     return

        # 因为是串行的，我们可以随机选任意一个目标，不需要担心它身上有没有任务
        target_uid = self.n_agents + self.spawned_tasks_count

        # 随机生成坐标
        self._add_node(target_uid, NodeType.TARGET, (0.0, 0.0, 0.0, 0.0), "Target", self.spawned_tasks_count)
        target = self.nodes[-1]  # 刚刚添加进去的节点

        # 2. 生成任务实体，绑定该目标 ID
        task = TaskEntity(
            task_id=f"Task-{self.current_step}-{self.spawned_tasks_count}",
            target_node_id=target.node_id,
            time_window=self.cfg.task_duration,
            total_time=self.cfg.task_duration,
            start_time=float(self.current_step)
        )

        self.tasks.append(task)
        self.spawned_tasks_count += 1

        # [日志] 任务生成
        # print(f"[Rank {self.rank}][Step {self.current_step}] 🟢 新任务生成: {task.task_id}", flush=True)

    def _print_tsdit_chain(self, task, pk, time, success, reason=""):
        """
        打印多跳杀伤链详细战报 (优化版：显示节点类型 S/D/I + 完整链路)
        """
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        CYAN = "\033[96m"
        END = "\033[0m"

        status_label = f"{GREEN}[成功闭环]{END}" if success else f"{RED}[打击失败 ({reason})]{END}"

        print(f"\n{BLUE}>>>>>> 杀伤链战报 (Step: {self.current_step}) <<<<<<{END}", flush=True)
        print(f"【任务ID】: {task.task_id}", flush=True)
        print(f"【状态】: {status_label}", flush=True)

        # 1. 辅助函数：根据ID获取带类型的名称 (如 S1, D8)
        def get_node_str(nid):
            if 0 <= nid < len(self.nodes):
                node = self.nodes[nid]

                # 如果是目标节点，依然保持 T0, T1 这种简洁打印
                if node.node_type == NodeType.TARGET:
                    return f"T{node.local_id}"

                # 如果是智能体，使用 "组名_编号" 格式
                # 例如：SensorA_2
                return f"{node.group_name}_{node.local_id}"

            return f"?{nid}"

        log = task.active_chain_log
        if log:

            # 2. 辅助函数：格式化路径
            def format_trace(trace, color=END):
                if not trace: return "Unknown"
                # 将 ID 列表转换为 "S1 -> S2" 格式
                names = [get_node_str(uid) for uid in trace]
                return " -> ".join([f"{color}{name}{END}" for name in names])

            # 获取记录的路径
            trace_s_d = log.get('trace_s_d', [])
            trace_d_i = log.get('trace_d_i', [])
            p_s_d = log.get('prob_s_d', 0.0)
            p_d_i = log.get('prob_d_i', 0.0)

            # 3. 打印物理链路详情 (分段显示)
            print(f"【物理链路】:", flush=True)

            # S -> D 段
            path_str_1 = format_trace(trace_s_d, CYAN)
            hops_1 = max(0, len(trace_s_d) - 1)
            print(f"  test-link(S->D): {path_str_1}", flush=True)
            print(f"    └─ [跳数: {hops_1}] [累积概率: {p_s_d:.4f}]", flush=True)

            # D -> I 段
            path_str_2 = format_trace(trace_d_i, YELLOW)
            hops_2 = max(0, len(trace_d_i) - 1)
            print(f"  test-link(D->I): {path_str_2}", flush=True)
            print(f"    └─ [跳数: {hops_2}] [累积概率: {p_d_i:.4f}]", flush=True)

            # 4. 打印核心参数
            print(f"【杀伤参数】: Pk={pk:.4f} | TimeCost={time:.1f}", flush=True)
            print(f"  - Q1 (探测): {log['probs'][0]:.2f}", flush=True)
            print(f"  - P2 (S->D): {p_s_d:.4f}", flush=True)
            print(f"  - Q3 (决策): {log['probs'][2]:.2f}", flush=True)
            print(f"  - P3 (D->I): {p_d_i:.4f}", flush=True)
            print(f"  - Q4 (打击): {log['probs'][3]:.2f}", flush=True)

            # 5. [新增] 打印完整杀伤闭环
            # 拼接逻辑: T -> S... -> D... -> I... -> T
            target_str = get_node_str(task.target_node_id)

            # 合并路径：trace_s_d 的尾部 和 trace_d_i 的头部通常是同一个 D 节点，需要去重
            full_chain_ids = []
            if trace_s_d:
                full_chain_ids.extend(trace_s_d)

            if trace_d_i:
                # 如果 trace_s_d 的最后一个元素 == trace_d_i 的第一个元素 (都是 D节点)，则跳过第一个
                start_idx = 1 if (full_chain_ids and full_chain_ids[-1] == trace_d_i[0]) else 0
                full_chain_ids.extend(trace_d_i[start_idx:])

            # 生成链路字符串
            chain_nodes_str = " -> ".join([get_node_str(uid) for uid in full_chain_ids])
            full_chain_str = f"{target_str} -> {chain_nodes_str} -> {target_str}"

            print(f"杀伤链路：{full_chain_str}", flush=True)


        # === 情况B: 中途失败/超时 (无日志) ===

        else:

            print(f"【当前阶段】: {task.stage.name}", flush=True)
            # 打印已知的节点分配情况
            # S列表
            s_list_str = ",".join([get_node_str(sid) for sid in task.assigned_s_list]) if task.assigned_s_list else "无"
            # D节点
            d_str = get_node_str(task.assigned_d) if task.assigned_d != -1 else "无"
            # I列表
            i_list_str = ",".join([get_node_str(iid) for iid in task.assigned_i_list]) if task.assigned_i_list else "无"

            print(f"【任务进度快照】:", flush=True)
            print(f"  - 发现者 (S): [{s_list_str}]", flush=True)
            print(f"  - 指控者 (D): [{d_str}]", flush=True)
            print(f"  - 打击者 (I): [{i_list_str}]", flush=True)

            # 简单的故障诊断
            print(f"【诊断分析】:", flush=True)
            if task.stage == TaskStage.PENDING:
                print(f"  ⚠️  目标从未被任何 Sensor 发现 (可能是 S 探测能力不足或覆盖死角)", flush=True)
            elif task.stage == TaskStage.DETECTED:
                print(f"  ⚠️  S 已发现目标，但情报尚未传达给 D。", flush=True)
                print(f"      可能原因: S->D 链路中断、跳数过多导致超时、或 D 节点 Buffer 已满。", flush=True)
            elif task.stage == TaskStage.ASSIGNED:
                print(f"  ⚠️  D 已收到情报，但指令尚未传达给 I。", flush=True)
                print(f"      可能原因: D 正在决策(计算时延)、D->I 链路中断、或没有 I 节点在通信范围内。", flush=True)
            elif task.stage == TaskStage.STRIKING:
                print(f"  ⚠️  I 已收到指令并开始打击，但武器飞行时间过长导致超时。", flush=True)
                print(f"      可能原因: I 距离目标太远、或武器飞行速度太慢。", flush=True)

        print(f"{BLUE}{'=' * 60}{END}\n", flush=True)

    def get_share_obs(self, obs_list):
        """
        紧凑 Global State:
        [agents(max_agents*8), stats(6), tasks(max_targets*5)]
        task feat = [valid, x, y, z, stage]
        """
        agent_feats = []
        for n in self.nodes[:self.n_agents]:
            speed_norm = max(self._get_node_speed_km_per_step(n), 1e-6)
            feat = [
                n.pos[0] / self.cfg.space_size_x,
                n.pos[1] / self.cfg.space_size_y,
                n.pos[2] / self.cfg.space_size_z,
                n.node_type.value / 3.0,
                1.0 if n.alive else 0.0,
                n.vel[0] / speed_norm,
                n.vel[1] / speed_norm,
                n.vel[2] / speed_norm
            ]
            agent_feats.extend(feat)

        padding_count = self.max_agents_limit - self.n_agents
        if padding_count > 0:
            agent_feats.extend([0.0] * (self.agent_share_feat_dim * padding_count))

        pending_count = sum(1 for t in self.tasks if t.stage == TaskStage.PENDING)
        detected_count = sum(1 for t in self.tasks if t.stage == TaskStage.DETECTED)
        assigned_count = sum(1 for t in self.tasks if t.stage in [TaskStage.ASSIGNED, TaskStage.STRIKING])
        completed_count = sum(1 for t in self.tasks if t.stage == TaskStage.COMPLETED)
        active_count = sum(1 for t in self.tasks if t.stage not in [TaskStage.COMPLETED, TaskStage.FAILED])

        denom = max(1.0, float(self.curr_n_target))
        stats_feat = np.array([
            self.curr_n_target / max(1.0, float(self.max_targets_limit)),
            active_count / denom,
            pending_count / denom,
            detected_count / denom,
            assigned_count / denom,
            completed_count / denom,
        ], dtype=np.float32)

        task_feats = np.zeros(self.max_targets_limit * 5, dtype=np.float32)
        sorted_tasks = sorted(self.tasks, key=lambda t: t.target_node_id)

        for t in sorted_tasks:
            slot = t.target_node_id - self.n_agents
            if slot < 0 or slot >= self.max_targets_limit:
                continue

            target_node = next((n for n in self.nodes if n.node_id == t.target_node_id), None)
            if target_node is None:
                continue

            t_feat = np.array([
                1.0,
                target_node.pos[0] / self.cfg.space_size_x,
                target_node.pos[1] / self.cfg.space_size_y,
                target_node.pos[2] / self.cfg.space_size_z,
                t.stage.value / 5.0
            ], dtype=np.float32)

            start = slot * 5
            task_feats[start:start + 5] = t_feat

        global_state = np.concatenate([
            np.array(agent_feats, dtype=np.float32),
            stats_feat,
            task_feats
        ]).astype(np.float32)

        return [global_state.copy() for _ in range(self.n_agents)]

    def get_available_actions(self):
        """
        根据当前状态生成可用动作 Mask。

        混合动作结构：
            [strategy, candidate_idx, vx, vy, vz]

        Mask 只覆盖离散部分：
            strategy_mask: 3 维
            candidate_mask: max_candidates 维

        连续速度 vx/vy/vz 不做可用动作 mask，由高斯动作头直接输出。
        """
        avail_actions = []

        for i in range(self.n_agents):
            agent = self.nodes[i]
            mask = np.zeros(3 + self.max_candidates, dtype=np.float32)

            # ===== dead agent: 只允许 no-op + candidate 0 =====
            if not agent.alive:
                mask[0] = 1.0
                mask[3 + 0] = 1.0
                avail_actions.append(mask)
                continue

            # ===== 1. strategy mask =====
            mask[0] = 1.0  # no-op

            if agent.node_type == NodeType.SENSOR:
                det_range, _, _ = agent.get_actual_ranges(self.cfg)
                detectable_targets = []
                for node in self.nodes:
                    if node.node_type == NodeType.TARGET and node.alive:
                        d = np.linalg.norm(agent.pos - node.pos)
                        if d <= det_range:
                            detectable_targets.append((d, node))
                detectable_targets.sort(key=lambda x: x[0])
                valid_detect_slots = min(len(detectable_targets), self.max_candidates)
                if valid_detect_slots > 0:
                    mask[1] = 1.0

            elif agent.node_type == NodeType.DECISION:
                if len(agent.data_buffer) > 0:
                    mask[1] = 1.0

            elif agent.node_type == NodeType.INTERCEPTOR:
                if len(agent.data_buffer) > 0:
                    mask[1] = 1.0

            if len(agent.tx_buffer) > 0:
                mask[2] = 1.0

            # ===== 2. candidate mask =====
            candidate_mask = np.zeros(self.max_candidates, dtype=np.float32)

            # sensor 的 produce 对应“目标候选”
            if agent.node_type == NodeType.SENSOR:
                det_range, _, _ = agent.get_actual_ranges(self.cfg)
                detectable_targets = []
                for node in self.nodes:
                    if node.node_type == NodeType.TARGET and node.alive:
                        d = np.linalg.norm(agent.pos - node.pos)
                        if d <= det_range:
                            detectable_targets.append((d, node))
                detectable_targets.sort(key=lambda x: x[0])
                valid_detect_slots = min(len(detectable_targets), self.max_candidates)
                candidate_mask[:valid_detect_slots] = 1.0

            # comm 对应“可通信邻居候选”
            _, a_comm, _ = agent.get_actual_ranges(self.cfg)
            neighbor_candidates = []
            for other in self.nodes:
                if other.node_id == agent.node_id:
                    continue
                d = np.linalg.norm(agent.pos - other.pos)
                link_prob = self._calc_q2_comm_prob(agent, other) if other.alive else 0.0
                if other.alive and d <= a_comm and link_prob > 0.25:
                    neighbor_candidates.append((d, other))
            neighbor_candidates.sort(key=lambda x: x[0])
            valid_neighbor_slots = min(len(neighbor_candidates), self.max_candidates)
            candidate_mask[:valid_neighbor_slots] = 1.0

            # D/I 的 strategy=1 当前不使用 candidate_idx，因此只要可 produce，就放开 candidate
            if (agent.node_type in [NodeType.DECISION, NodeType.INTERCEPTOR]) and len(agent.data_buffer) > 0:
                candidate_mask[:] = 1.0

            # 避免 candidate 头出现全 0，因为 categorical 采样必须至少有一个合法动作。
            if candidate_mask.sum() <= 0:
                candidate_mask[0] = 1.0

            mask[3:3 + self.max_candidates] = candidate_mask
            avail_actions.append(mask)

        return avail_actions

    def seed(self, seed=None):
        np.random.seed(seed)

    def close(self):
        pass

    def render(self, mode='human'):
        pass



    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)

    def close(self):
        """关闭环境"""
        pass

    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            print(f"\n===== Step {self.current_step} =====")
            print(f"Active Tasks: {len([t for t in self.tasks if t.stage != TaskStage.COMPLETED])}")
            print(f"Completed Tasks: {self.completed_tasks}")
            print(f"Alive Agents: {len([n for n in self.nodes[:self.n_agents] if n.alive])}")
            print(f"Messages in Queue: {len(self.message_queue)}")