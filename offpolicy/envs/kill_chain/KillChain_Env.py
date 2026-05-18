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

    def __init__(self, args, rank=0):
        """初始化环境"""
        super(KillChainEnv, self).__init__()
        self.rank = rank
        self.args = args
        self.cfg = Config()

        self.grid_res = np.array([
            self.cfg.space_size_x / self.cfg.grid_size_x,
            self.cfg.space_size_y / self.cfg.grid_size_y,
            self.cfg.space_size_z / self.cfg.grid_size_z
        ], dtype=np.float32)

        self.map_size = np.array([
            self.cfg.space_size_x,
            self.cfg.space_size_y,
            self.cfg.space_size_z
        ], dtype=np.float32)
        # 定义距离归一化的参考标尺 (通常取最长边)
        self.max_span = max(self.cfg.space_size_x, self.cfg.space_size_y, self.cfg.space_size_z)

        # 基本参数
        self.n_agents = self.cfg.n_agents
        self.n_total = self.cfg.n_total_nodes
        self.max_steps = self.args.episode_length
        self.current_step = 0

        self.max_agents_limit = self.cfg.max_agents_limit
        self.max_targets_limit = self.cfg.max_targets_limit

        # 状态变量
        self.nodes: List[EndogenousNode] = []
        self.tasks: List[TaskEntity] = []
        self.interaction_history: List[Interaction] = []
        self.message_queue: List[Message] = []
        self.msg_counter = 0

        # 动作空间
        # Dim 0: Strategy (0:Comm, 1:Attack, 2:Detect) - 假设原本是这三个
        # Dim 1: Candidate (选择交互对象)
        # Dim 2: Movement (0:Stay, 1:X+, 2:X-, 3:Y+, 4:Y-, 5:Z+, 6:Z-)
        self.max_candidates = 5
        self.task_feat_dim = 6
        self.action_space = []
        for _ in range(self.n_agents):
            act_space = spaces.MultiDiscrete([3, self.max_candidates, 7])
            self.action_space.append(act_space)

        # ==========================
        # 4. 观测空间定义 (Observation Space)
        # ==========================
        # 基础Obs(12) + Buffer特征(4) + 任务(5) + 邻居(10*7) = 91

        self.buffer_feat_dim = 7  # [新增] Buffer特征维度
        self.compass_dim = 6  # [RelX_Mate, RelY_Mate, RelZ_Mate, RelX_Target, RelY_Target, RelZ_Target]
        self.max_observed_tasks = self.max_targets_limit
        self.obs_dim = (10 +
                        self.buffer_feat_dim +
                        self.compass_dim +
                        (self.task_feat_dim * self.max_observed_tasks) +
                        (self.max_candidates * 7))

        # print(f"!!! DEBUG !!! KillChainEnv 初始化: obs_dim = {self.obs_dim}", flush=True)

        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # self.share_obs_dim = (8 * self.n_agents) + 5
        # self.share_obs_dim = self.obs_dim * self.n_agents
        self.share_obs_dim = (self.max_agents_limit * 5) + (self.max_targets_limit * 4)

        # 打印调试信息，确认计算出的维度是否为 167 (18*9 + 5 = 167)
        # print(f"[KillChainEnv] Obs_Dim: {self.obs_dim}, Share_Obs_Dim: {self.share_obs_dim}")

        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # 统计信息
        self.total_reward = 0.0
        self.cooperation_count = 0.0
        self.completed_tasks = 0.0

    def reset(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """重置环境，必须返回 3 个值"""
        self.current_step = 0
        self.total_reward = 0.0
        self.occupied_positions = set()  # 占用表，防止初始生成重叠
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

        self._initialize_nodes()

        self.target_nodes_list = [n for n in self.nodes if n.node_type == NodeType.TARGET]
        self.max_mission_count = self.cfg.n_target
        self.spawned_tasks_count = 0
        self.completed_tasks = 0.0

        self.tasks = []
        for i in range(self.cfg.n_target):
            # 目标 ID 紧接着智能体 ID
            target_uid = self.n_agents + i
            self._add_node(target_uid, NodeType.TARGET, (0.0, 0.0, 0.0, 0.0), "Target", i)
            task = TaskEntity(
                task_id=f"Task-0-{i}",  # ID 格式: Task-Step-Index
                target_node_id=target_uid,
                time_window=self.cfg.task_duration,
                total_time=self.cfg.task_duration,
                start_time=0.0
            )
            task.stage = TaskStage.PENDING  # 初始状态：潜伏
            self.tasks.append(task)
        # for i in range(self.cfg.n_target):
        #     # 目标 ID 紧接着智能体 ID
        #     target_uid = self.n_agents + i
        #     # 调用你之前改好的 _add_node (会自动把目标放在外部区域)
        #     self._add_node(target_uid, NodeType.TARGET, (0.0, 0.0, 0.0, 0.0), "Target", i)

        # 1. 获取基础观测
        obs = self.get_obs()
        # 2. 生成共享观测
        share_obs = self.get_share_obs(obs)
        # 3. 生成动作Mask
        available_actions = self.get_available_actions()

        # [修复] 返回 3 个值以匹配 Wrapper
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
        [修改] 简化的分布初始化
        - S/D/I: 统一在中心 60% 区域随机分布 (内线防御)
        - Target: 在外部区域随机分布 (外线进攻)
        """
        max_attempts = 1000

        Gx = self.cfg.grid_size_x
        Gy = self.cfg.grid_size_y
        Gz = self.cfg.grid_size_z

        # 定义中心 60% 区域的边界
        # 假设地图范围 [0, G], 中心 G/2
        # 60% 区域意味着左右各留白 20%
        margin_rate = 0.2
        x_min_in = int(margin_rate * Gx)
        x_max_in = int((1 - margin_rate) * Gx)
        y_min_in = int(margin_rate * Gy)
        y_max_in = int((1 - margin_rate) * Gy)

        for _ in range(max_attempts):

            # === S/D/I: 内部随机 ===
            if ntype in [NodeType.SENSOR, NodeType.DECISION, NodeType.INTERCEPTOR]:
                gx = np.random.randint(x_min_in, x_max_in)
                gy = np.random.randint(y_min_in, y_max_in)

                # Z轴处理：防空单元通常在地面部署
                # 设为 0 可以减少 Z 轴带来的额外搜索难度，让智能体专注于平面组网
                gz = 0
                # 如果您希望它们也能立体部署(如预警机)，可以改为: np.random.randint(0, int(0.2*Gz))

            # === Target: 外部随机 ===
            elif ntype == NodeType.TARGET:
                # 采用拒绝采样法：在全图生成，如果落在内部区域则重随
                # 外部区域面积占比约 64%，拒绝采样效率很高
                while True:
                    gx = np.random.randint(0, Gx)
                    gy = np.random.randint(0, Gy)

                    # 如果落在内部区域 (x_min <= x < x_max 且 y_min <= y < y_max)，则重来
                    if (x_min_in <= gx < x_max_in) and (y_min_in <= gy < y_max_in):
                        continue
                    else:
                        break

                # Z轴处理：目标通常在空中
                # 设定在较低高度以上，避免贴地飞行导致探测困难
                gz = np.random.randint(int(0.2 * Gz), Gz)

            else:
                # Fallback
                gx = np.random.randint(0, Gx)
                gy = np.random.randint(0, Gy)
                gz = np.random.randint(0, Gz)

            # === 占用检查与实例化 ===
            candidate_pos = (gx, gy, gz)

            if candidate_pos not in self.occupied_positions:
                self.occupied_positions.add(candidate_pos)

                node = EndogenousNode(
                    node_id=uid,
                    node_type=ntype,
                    pos=np.zeros(3, dtype=np.float32),
                    grid_pos=np.array([gx, gy, gz], dtype=int),
                    energy=self.cfg.init_energy,
                    raw_capability=np.array(cap_base, dtype=np.float32),
                    group_name=group_name,
                    local_id=local_id
                )

                # 同步物理坐标
                node.update_position_from_grid(np.array(candidate_pos), self.grid_res)
                self.nodes.append(node)
                return

        print(f"Warning: Grid full! Failed to spawn node {uid}")

    def step(self, actions: List[np.ndarray]):
        """执行一步环境交互，必须返回 6 个值"""
        self.current_step += 1
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        dones = np.zeros(self.n_agents, dtype=bool)

        # 1. 记录移动前的位置 (用于计算 Shaping Reward)
        old_positions = [n.pos.copy() for n in self.nodes]

        # 2. 处理物理移动 (撞墙/撞人惩罚在这里产生)
        move_rewards = self._process_movement(actions)
        rewards += move_rewards

        # 3. 计算 Buffer 负载 (统计指标)
        current_loads = []
        for agent in self.nodes[:self.n_agents]:
            if agent.alive:
                # 根据类型确定容量
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

        # =================================================================
        # [核心修改] 4. 距离引导奖励 (Distance Shaping Reward)
        # =================================================================
        # 替代了原有的 "逻辑A/逻辑B"
        for i, agent in enumerate(self.nodes[:self.n_agents]):
            if not agent.alive:
                continue

            # 获取上一帧位置
            old_pos = old_positions[i] if i < len(old_positions) else agent.pos

            # --- 情况 A: Interceptor (拦截器) ---
            # 只有当它肚子里有指令时，才引导它去打击
            if agent.node_type == NodeType.INTERCEPTOR:
                target_id = -1
                # 检查 Buffer 第一条是否为有效指令
                if len(agent.data_buffer) > 0:
                    msg = agent.data_buffer[0]
                    if isinstance(msg, Message) and 'target_id' in msg.content:
                        target_id = msg.content['target_id']

                if target_id != -1:
                    target = next((n for n in self.nodes if n.node_id == target_id), None)
                    # 只有目标还活着才引导
                    if target and target.alive:
                        curr_dist = np.linalg.norm(agent.pos - target.pos)
                        old_dist = np.linalg.norm(old_pos - target.pos)

                        # [引导公式] 靠近给正分，远离给负分
                        # 假设 shaping_factor = 2.0, 移动一步约 0.1km -> 奖励 +0.2
                        delta = old_dist - curr_dist
                        rewards[i] += self.cfg.shaping_factor * delta

                        # [锁定奖励] 如果已经进入射程，给予持续的小奖励，防止它进进出出
                        _, _, strike_range = agent.get_actual_ranges(self.cfg)
                        if curr_dist <= strike_range:
                            rewards[i] += 0.05

            # --- 情况 B: Sensor (传感器) ---
            # 简单的巡逻/追踪引导：如果最近探测过某个目标，鼓励保持在探测范围内
            elif agent.node_type == NodeType.SENSOR:
                # 寻找最近的已知/存活目标
                closest_dist = float('inf')
                closest_target = None
                det_range, _, _ = agent.get_actual_ranges(self.cfg)

                for t_node in self.target_nodes_list:
                    if t_node.alive:
                        d = np.linalg.norm(agent.pos - t_node.pos)
                        if d < closest_dist:
                            closest_dist = d
                            closest_target = t_node

                # 如果这个目标在 "稍微有点远但能感知" 的范围内，鼓励靠近
                # 这里给一个较小的引导，避免 S 节点无脑冲脸送死
                if closest_target and closest_dist < det_range * 2.0:
                    old_dist = np.linalg.norm(old_pos - closest_target.pos)
                    delta = old_dist - closest_dist

                    # 只有还没贴脸时才奖励 (保持 0.5 倍射程的安全距离)
                    if closest_dist > det_range * 0.5:
                        rewards[i] += (self.cfg.shaping_factor * 0.5) * delta

        # 5. 基础能耗扣除
        for i, node in enumerate(self.nodes):
            if node.alive:
                node.consume_energy(self.cfg.energy_cost_hover)
                # 死亡判定 (可选)
                if not node.alive:
                    rewards[i] -= 1.0

        # 6. 更新任务时间
        self._update_tasks()

        # 7. 处理动作交互 (探测、通信、打击指令生成)
        self.interaction_history = []
        self._process_actions(actions, rewards)

        # 8. 处理消息到达 (路由与中继奖励在这里产生)
        self._process_arrived_messages(rewards)

        # 9. 执行杀伤链闭环逻辑 (打击判定、任务完成/失败结算)
        step_rewards = self._execute_kill_chain()
        rewards += step_rewards

        # 10. 全局 Pending 惩罚 (建议在 config 中设为 0 或极小值 0.1)
        pending_tasks_count = len([t for t in self.tasks if t.stage == TaskStage.PENDING])
        if pending_tasks_count > 0:
            total_penalty = pending_tasks_count * self.cfg.pending_penalty_per_task
            for i in range(self.n_agents):
                if self.nodes[i].alive:
                    rewards[i] -= total_penalty

        # 11. 统计与分类型奖励记录
        for i in range(self.n_agents):
            agent_type = self.nodes[i].node_type
            self.type_episode_rewards[agent_type] += rewards[i]

        self.total_reward += np.sum(rewards)

        # 12. 观测与结束条件
        obs = self.get_obs()
        share_obs = self.get_share_obs(obs)
        available_actions = self.get_available_actions()

        is_time_limit = (self.current_step >= self.max_steps)
        is_mission_cleared = (self.completed_tasks >= self.max_mission_count)

        if is_time_limit or is_mission_cleared:
            dones = np.array([True] * self.n_agents)

            # --- 统计指标计算 (保持原样) ---
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

            info = {
                "kill_chain/total_reward": self.total_reward,
                "kill_chain/cooperation_count": self.cooperation_count,
                "kill_chain/completed_tasks": self.completed_tasks,
                "kill_chain/success_rate": success_rate,
                "kill_chain/is_success": 1.0 if is_mission_cleared else 0.0,
                "kill_chain/avg_pk": avg_pk,
                "kill_chain/avg_time_eff": avg_time_eff,
                "kill_chain/paper_temporal_efficiency": avg_paper_eff,
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

        # 13. 清理已完成任务的 Buffer (Buffer GC)
        self._clean_completed_tasks_from_buffers()

        return obs, share_obs, rewards, dones, infos, available_actions

    def _process_movement(self, actions):
        """
        处理所有智能体的移动指令
        Action Dim 2 definition:
        0: Stay
        1: X+ (East)
        2: X- (West)
        3: Y+ (South)
        4: Y- (North)
        5: Z+ (Up)
        6: Z- (Down)
        """
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        # 1. 构建当前时刻的占用地图 (Set of tuples)
        #    用于快速查找某个格子是否有人
        occupied = set()
        for node in self.nodes:
            if node.alive:
                occupied.add(tuple(node.grid_pos))

        # 2. 定义 7 个动作对应的坐标增量 (dx, dy, dz)
        deltas = [
            [0, 0, 0],  # 0: Stay
            [1, 0, 0],  # 1: X+
            [-1, 0, 0],  # 2: X-
            [0, 1, 0],  # 3: Y+
            [0, -1, 0],  # 4: Y-
            [0, 0, 1],  # 5: Z+
            [0, 0, -1]  # 6: Z-
        ]

        agent_indices = np.arange(self.n_agents)
        np.random.shuffle(agent_indices)

        # 3. 按随机顺序遍历智能体执行移动
        for i in agent_indices:
            agent = self.nodes[i]

            # 死人跳过
            if not agent.alive:
                continue

            # 获取该智能体的动作
            move_cmd = int(actions[i][2])

            if move_cmd == 0:
                continue

            # 计算预期的新网格坐标
            delta = np.array(deltas[move_cmd])
            current_g = agent.grid_pos
            new_g = current_g + delta

            # --- 检查 A: 越界 (Boundary Check) ---
            if (new_g[0] < 0 or new_g[0] >= self.cfg.grid_size_x or
                    new_g[1] < 0 or new_g[1] >= self.cfg.grid_size_y or
                    new_g[2] < 0 or new_g[2] >= self.cfg.grid_size_z):
                # 惩罚：撞墙
                rewards[i] -= 2
                agent.consume_energy(self.cfg.energy_cost_collision)
                continue

            # --- 检查 B: 碰撞 (Collision Check) ---
            if tuple(new_g) in occupied:
                # 惩罚：与其他智能体相撞 (或者撞到原本就在那里的静态节点)
                rewards[i] -= 2
                agent.consume_energy(self.cfg.energy_cost_collision)
                continue

            # --- C. 移动成功 (Move Success) ---
            # 1. 更新占用表
            occupied.remove(tuple(current_g))
            occupied.add(tuple(new_g))

            # 2. 更新坐标
            agent.update_position_from_grid(new_g, self.grid_res)

            # 3. 扣除移动能耗
            agent.consume_energy(self.cfg.energy_cost_move)

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
        处理所有智能体的动作：解析意图并验证物理约束 (多跳盲路由完整版)

        逻辑分层:
        1. 状态更新: 模式切换 (IDLE/NORMAL/HIGH)
        2. 策略执行:
           - Strategy 1 (Produce): 生成消息放入 TX Buffer (S探测/D决策/I打击)
           - Strategy 2 (Comm): 从 TX Buffer 取出消息发送给邻居 (Blind Routing)
        """

        # 初始化本帧的通道占用计数器 (暂未严格限制，可用于扩展)
        current_channel_usage = {i: 0 for i in range(self.n_agents)}

        for agent_id, action in enumerate(actions):
            agent = self.nodes[agent_id]

            # 1. 存活检查
            if not agent.alive:
                continue

            # A. 动作解析
            strategy = int(action[0])  # 0=No-Op, 1=Produce/Execute, 2=Comm/Relay
            candidate_idx = int(action[1])  # 邻居/目标 索引 (0~K-1)

            # 动态获取当前能力值比例下的实际物理半径
            a_det, a_comm, a_str = agent.get_actual_ranges(self.cfg)

            # 策略 0: No-Op
            if strategy == 0:
                continue

            # ===============================================================
            # 策略 1: 生产与执行 (Produce & Execute)
            # S: 探测并生成情报 -> TX Buffer
            # D: 处理情报生成指令 -> TX Buffer
            # I: 消耗指令执行打击
            # ===============================================================
            if strategy == 1:

                # --- SENSOR: 探测目标并生成情报 ---
                # --- SENSOR: 探测目标并生成情报 ---
                if agent.node_type == NodeType.SENSOR:
                    # 1. 获取所有存活的目标节点
                    potential_targets = []
                    for node in self.nodes:
                        if node.node_type == NodeType.TARGET and node.alive:
                            dist = np.linalg.norm(agent.pos - node.pos)
                            potential_targets.append((dist, node))
                    # 按距离排序
                    potential_targets.sort(key=lambda x: x[0])

                    # 2. 尝试探测选定的目标
                    if candidate_idx < len(potential_targets):
                        target_dist, target_node = potential_targets[candidate_idx]

                        # 物理探测半径检查
                        if target_dist <= a_det:
                            # [回滚] 简单粗暴给奖励：只要探测到就给钱
                            if target_node.node_id not in self.detected_targets_history:
                                rewards[agent_id] += self.cfg.detection_reward
                                self.detected_targets_history.add(target_node.node_id)
                            else:
                                # 已经发现过的目标，只给极少的跟踪奖励，或者不给
                                rewards[agent_id] += 0.05

                            # 记录探测交互 (可视化用)
                            interaction = Interaction(agent_id, target_node.node_id, NodeType.SENSOR, NodeType.TARGET,
                                                      target_dist, "detect", None)
                            self.interaction_history.append(interaction)

                            # 扣除探测能耗
                            agent.consume_energy(self.cfg.energy_cost_detection)
                            # 检查该目标是否已经关联了任务
                            task_id = None
                            target_task = None

                            for t in self.tasks:
                                if t.target_node_id == target_node.node_id and t.stage != TaskStage.COMPLETED:
                                    task_id = t.task_id
                                    target_task = t
                                    break
                            if target_task is not None:
                                # 如果任务还是 PENDING (潜伏状态)，说明是【首次发现】
                                if target_task.stage == TaskStage.PENDING:
                                    target_task.stage = TaskStage.DETECTED
                                    # 记录发现时间戳
                                    target_task.first_detection_time = float(self.current_step)

                            # if target_task is None:
                            #     # 生成新的 TaskEntity
                            #     new_task_id = f"Task-{self.current_step}-{target_node.local_id}"
                            #
                            #     new_task = TaskEntity(
                            #         task_id=new_task_id,
                            #         target_node_id=target_node.node_id,
                            #         time_window=self.cfg.task_duration,
                            #         total_time=self.cfg.task_duration,
                            #         start_time=float(self.current_step)
                            #     )
                            #     # 标记为 DETECTED (因为刚被发现)
                            #     new_task.stage = TaskStage.DETECTED
                            #     new_task.first_detection_time = float(self.current_step)
                            #
                            #     # 加入全局任务列表
                            #     self.tasks.append(new_task)
                            #
                            #     # 更新局部变量，供下面生成情报使用
                            #     task_id = new_task_id
                            #     target_task = new_task
                            #
                            #     # print(f"🟢 [Step {self.current_step}] S{agent.local_id} 激活新目标: {target_node.local_id}"

                            # [核心逻辑] 生成情报消息，存入 TX Buffer
                            if len(agent.tx_buffer) < 20:  # 缓冲区容量限制
                                # 计算探测置信度 Q1
                                q1 = self._calc_q1_detect_prob(agent, target_node)

                                if task_id:
                                    # [Buffer去重]
                                    if target_task.first_detection_time is None:
                                        target_task.first_detection_time = float(self.current_step)
                                    # 即使允许刷分，Buffer里也没必要塞满完全一样的重复情报，检查一下去重
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
                                                "s_reporter_id": agent_id
                                            },
                                            send_time=0, arrival_time=0,
                                            path_prob=1.0
                                        )
                                        agent.tx_buffer.append(msg)
                                        self.msg_counter += 1
                                    else:
                                        pass
                # --- DECISION: 处理情报并生成指令 ---
                elif agent.node_type == NodeType.DECISION:
                    # 只有当 Buffer 有情报时才能决策
                    if len(agent.data_buffer) > 0:
                        # 1. 读取情报
                        intel_data = agent.data_buffer.pop(0)

                        task_id = intel_data['task_id']
                        task = next((t for t in self.tasks if t.task_id == task_id), None)

                        if task and task.stage != TaskStage.COMPLETED:
                            # [D端去重] 检查发送缓冲区是否已有针对该任务的指令
                            has_cmd_in_buffer = False
                            for pending_msg in agent.tx_buffer:
                                if (pending_msg.msg_type == "strike_command" and
                                        pending_msg.content['task_id'] == task_id):
                                    has_cmd_in_buffer = True
                                    break

                            if not has_cmd_in_buffer:
                                task.assigned_d = agent_id
                                # 计算 Q3 (决策质量)
                                q3 = self._calc_q3_command_prob(agent, 1.0)
                                # 生成指令消息
                                msg = Message(
                                    msg_id=f"cmd_{self.msg_counter}",
                                    initial_source_id=agent_id,  # 标记发起者 D
                                    from_id=agent_id,
                                    to_id=-1,  # 暂时未知，由 Strategy=2 (Comm) 决定
                                    target_node_type=NodeType.INTERCEPTOR,  # 目标类型：找 I
                                    msg_type="strike_command",
                                    content={
                                        "task_id": task_id,
                                        "target_id": task.target_node_id,
                                        "origin_confidence": intel_data['confidence'],
                                        # 继承 S->D 链路信息
                                        "accumulated_comm_prob_s_d": intel_data.get(
                                            'accumulated_comm_prob', 1.0),
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
                                    # 决策消耗 (算力消耗)
                                    agent.consume_energy(self.cfg.energy_cost_comm)
                            else:
                                # 重复指令，跳过生成
                                pass

                # --- INTERCEPTOR: 执行打击 ---
                elif agent.node_type == NodeType.INTERCEPTOR:
                    # 前置检查：只有 Buffer 里有指令才能执行
                    if len(agent.data_buffer) > 0:
                        selected_msg = agent.data_buffer.pop(0)

                        # === 执行打击逻辑 ===
                        if selected_msg:
                            rewards[agent_id] += self.cfg.strike_initiation_reward

                            # 调用核心打击结算 (计算 Pk, 判定是否击杀)
                            is_final = self._perform_strike_logic(agent, selected_msg)

                            # 3. [关键修改] 如果不是终局 (Soft Fail)，把指令塞回去！
                            if not is_final:
                                # 将指令插回队头，确保下一回合还能拿到
                                agent.data_buffer.insert(0, selected_msg)

                            # 消耗能量
                            agent.consume_energy(self.cfg.energy_cost_strike)
            elif strategy == 2:
                # 前置检查：只有当发送缓冲区有消息时，Comm 动作才有效
                if len(agent.tx_buffer) > 0:

                    # 1. 动作解析
                    # candidate_idx 现在代表“Obs里第 k 近的邻居” (Slot Index)
                    # 范围是 0 ~ (max_candidates-1)
                    target_neighbor_slot = candidate_idx

                    # 2. 复现 Obs 的邻居排序逻辑 (核心！)
                    # 必须重新计算“谁是第 k 个邻居”，以确保 Action 和 Obs 对齐
                    # -------------------------------------------------
                    neighbor_candidates = []
                    for other in self.nodes:
                        # 排除自己
                        if other.node_id == agent.node_id:
                            continue
                            # 计算距离 (使用 np.linalg.norm 计算欧氏距离)
                        d = np.linalg.norm(agent.pos - other.pos)
                        neighbor_candidates.append((d, other))
                    # 按距离从小到大排序 (与 get_obs 保持一致)
                    neighbor_candidates.sort(key=lambda x: x[0])

                    # 3. 执行路由逻辑
                    # 检查 A: 槽位是否有效 (没有越界)
                    if target_neighbor_slot < len(
                            neighbor_candidates) and target_neighbor_slot < self.max_candidates:

                        # 取出目标节点对象
                        target_dist, target_node = neighbor_candidates[target_neighbor_slot]
                        # 检查 B: 物理通信限制
                        # 即使在Obs里看到了它，也必须在通信半径内才能发送 (且目标必须活着)
                        _, a_comm, _ = agent.get_actual_ranges(self.cfg)
                        if target_node.alive:
                            link_prob = self._calc_q2_comm_prob(agent, target_node)
                        else:
                            link_prob = 0.0

                        # 2. 增加干扰阻断判定 (link_prob > 0.001)
                        # 逻辑：在通信范围内 + 目标活着 + 且信号没有被干扰到"几乎为0"
                        if target_dist <= a_comm and target_node.alive and link_prob > 0.25:
                            # === 发送执行 ===
                            msg = agent.tx_buffer.pop(0)  # 取出队头消息
                            # 检查 C: 防环路 (Loop Check)
                            # 如果目标曾经经手过这条消息，那就是发回去了，形成环路 -> 拒绝并惩罚
                            if target_node.node_id in msg.route_trace:
                                # 这是一个坏的路由选择
                                agent.consume_energy(self.cfg.energy_cost_comm)
                                # (可选) 可以在这里给一点微小的负奖励，教导它不要回传
                                # rewards[agent_id] -= 0.1
                            else:
                                # === 发送成功 (Success) ===
                                # 1. 计算物理层参数
                                delay = self._calculate_comm_delay(agent, target_node, target_dist)

                                # 2. 更新消息头
                                msg.from_id = agent.node_id
                                msg.to_id = target_node.node_id  # 下一跳是谁

                                # 3. 更新链路追踪信息
                                msg.path_prob *= link_prob  # 累乘链路概率
                                msg.hop_count += 1  # 跳数 +1
                                msg.route_trace.append(agent.node_id)  # 把自己签在路径本上

                                # 4. 设定时间戳
                                msg.send_time = self.current_step
                                msg.arrival_time = self.current_step + delay

                                # 5. 放入全局队列 (飞鸽传书)
                                self.message_queue.append(msg)

                                # 6. 记录交互历史 & 消耗能量
                                agent.consume_energy(self.cfg.energy_cost_comm)

                                interaction = Interaction(
                                    agent_id, target_node.node_id,
                                    agent.node_type, target_node.node_type,
                                    target_dist, "comm", None
                                )
                                self.interaction_history.append(interaction)

                        else:
                            # 失败分支 1: 目标太远(超出半径) 或 已阵亡
                            # 这是一个无效的物理尝试，扣除少量能量作为“尝试成本”
                            agent.consume_energy(self.cfg.energy_cost_comm * 0.2)

                    else:
                        # 失败分支 2: 索引越界
                        # 例如周围只有2个邻居，Action却选了Slot 4 (空空气)
                        # 这是一个无效的逻辑动作，通常直接跳过，或者给微小惩罚
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
            # 1. 将执行打击的 I 节点加入任务的参与者列表
            if i_node.node_id not in task.assigned_i_list:
                task.assigned_i_list.append(i_node.node_id)

            if task.stage != TaskStage.COMPLETED and task.stage != TaskStage.FAILED:
                task.set_stage(TaskStage.STRIKING, self.current_step)

            # 记录打击参与者
            if i_node.node_id not in task.assigned_i_list:
                task.assigned_i_list.append(i_node.node_id)

            target_node = next((n for n in self.nodes if n.node_id == task.target_node_id), None)

            # 1. 提取全链路概率参数
            # p1: 原始探测置信度 (S节点)
            p1 = msg.content.get("origin_confidence", 0.0)

            # p2: S->D 链路累积概率 (修复点：使用新键名 accumulated_comm_prob_s_d)
            p2 = msg.content.get("accumulated_comm_prob_s_d", msg.content.get("origin_comm_s_d", 1.0))

            # p3: 决策质量 Q3 (修复点：使用新键名 q3_prob)
            p3 = msg.content.get("q3_prob", msg.content.get("q3_command_prob", 1.0))

            # p4: 打击概率 Q4 (I节点本地计算)
            p4 = self._calc_q4_strike_prob(i_node, target_node)

            # p_link_2: D->I 的链路概率 (这是 msg 对象自带的 path_prob，经过多跳累积)
            p_link_2 = msg.path_prob

            # 2. 最终杀伤链概率 Pk 计算
            # 公式: Pk = Q1 * P(S->D) * Q3 * P(D->I) * Q4
            pk = p1 * p2 * p3 * p_link_2 * p4
            self.episode_pks.append(pk)
            # 获取两段路径
            trace_s_d = msg.content.get('s_d_trace', [])
            # msg.route_trace 是 D->...->I(前一跳)，加上 i_node.node_id 是完整路径
            trace_d_i = msg.route_trace + [i_node.node_id]

            # 记录战报
            task.active_chain_log = {
                's': msg.content['s_reporter_id'],
                'd': task.assigned_d,  # 或者从 msg.initial_source_id 获取
                'i': i_node.node_id,
                'probs': [p1, p2, p3, p4],  # p2是S->D累积，p4是打击，还需增加中间的指令链路概率

                # 保存路径数据
                'trace_s_d': trace_s_d,
                'trace_d_i': trace_d_i,
                'prob_s_d': p2,
                'prob_d_i': msg.path_prob
            }
            task.final_kill_prob = pk

            flight = self._calculate_flight_time(i_node, target_node)
            arrival_time = self.current_step + flight
            # 计算闭环时间 (Current + Flight - Start)
            # time_cost = arrival_time - task.start_time
            start_t = task.first_detection_time if task.first_detection_time is not None else task.start_time
            time_cost = arrival_time - start_t
            time_cost = max(0.0, time_cost)

            paper_eff = 1.0 / max(1.0, time_cost)
            self.episode_paper_effs.append(paper_eff)  # <--- 存入列表

            if task.stage != TaskStage.COMPLETED:
                if p4 >= self.cfg.min_kill_probability:
                    task.stage = TaskStage.COMPLETED
                    task.completion_time = arrival_time
                    self.completed_tasks += 1
                    self.cooperation_count += 1
                    # self._print_tsdit_chain(task, pk, arrival_time - task.start_time, True)

                    # 奖励结算 (只奖励该单链上的节点)
                    contributors = [msg.content['s_reporter_id'], task.assigned_d, i_node.node_id]
                    contributors = [c for c in contributors if c < self.n_agents]
                    if contributors:
                        base = task.accumulated_value + self.cfg.finish_const_reward
                        # 1. 获取任务的最大生命周期 (例如 50步)
                        max_window = self.cfg.task_duration
                        # 2. 计算时间效率 (Time Efficiency)
                        # 逻辑: 耗时越少，得分越高。如果刚好卡在超时边缘完成，得分接近 0。
                        # 公式: (最大时间 - 实际耗时) / 最大时间
                        # 范围: [0.0, 1.0]
                        time_eff = max(0.0, (max_window - time_cost) / max_window)
                        self.episode_time_effs.append(time_eff)

                        # total = base * pk * (0.5 + 0.5 * time_eff)
                        total = base * pk * (0.7 + 0.3 * time_eff)
                        task.pending_reward_distribution = (contributors, total / len(contributors))

                # === 分支 B: 打击失败 (概率不足，直接判负) ===
                else:
                    penalty = self.cfg.soft_failure_penalty

                    # 将惩罚分摊给贡献者 (或者只罚 I 节点也可以，视你的偏好)
                    # contributors = [msg.content['s_reporter_id'], task.assigned_d, i_node.node_id]
                    # contributors = [c for c in contributors if c < self.n_agents]
                    # if contributors:
                    #     share = penalty / len(contributors)
                    #     # 直接加到 pending 或者当前 reward (这里用 pending 统一处理)
                    #     task.pending_reward_distribution = (contributors, share)

                    contributors = [i_node.node_id]

                    # 过滤一下防止 ID 越界（虽然通常不会）
                    contributors = [c for c in contributors if c < self.n_agents]

                    if contributors:
                        # 此时 share = penalty (因为 len 只有 1)
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
        [修改版] get_obs: 支持固定维度的填充 (Padding)
        无论当前有多少个任务，输出的 task_state_flat 长度永远固定为 max_targets_limit * 6
        """
        obs_list = []
        K_NEIGHBORS = self.max_candidates
        NEIGHBOR_DIM = 7

        # --- 1. 预处理：批量提取所有节点数据 ---
        # 转换为 numpy 数组以便进行矩阵运算
        # all_pos: (N_total, 3)
        all_pos = np.array([n.pos for n in self.nodes], dtype=np.float32)
        all_alive = np.array([1.0 if n.alive else 0.0 for n in self.nodes], dtype=np.float32)
        all_types = np.array([n.node_type.value / 3.0 for n in self.nodes], dtype=np.float32)
        all_loads = np.array([n.load for n in self.nodes], dtype=np.float32)

        # 智能体的索引范围
        agent_indices = np.arange(self.n_agents)

        # --- 2. 批量计算距离矩阵 ---
        # agent_pos: (N_agents, 3)
        agent_pos = all_pos[:self.n_agents]

        # 扩展维度进行广播: (N_agents, 1, 3) - (1, N_total, 3)
        diff_matrix = agent_pos[:, np.newaxis, :] - all_pos[np.newaxis, :, :]
        # 距离矩阵: (N_agents, N_total)
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)

        # 归一化位置差分与距离
        diff_matrix_norm = (diff_matrix / self.map_size).astype(np.float32)
        dist_matrix_norm = (dist_matrix / self.max_span).astype(np.float32)

        # ==================== [核心修改] 任务特征填充逻辑 ====================

        # 1. 初始化全零向量，长度基于【最大限制】(max_targets_limit)
        # 这样无论真实任务有 4 个还是 12 个，容器永远是 12 个任务的大小
        # 这里的 self.max_observed_tasks 在 __init__ 里已经被设为了 self.cfg.max_targets_limit
        task_state_flat = np.zeros(self.max_observed_tasks * self.task_feat_dim, dtype=np.float32)

        # 2. 筛选可见任务 (排除已完成/失败的任务)
        visible_tasks = [
            t for t in self.tasks
            if t.stage not in [TaskStage.COMPLETED, TaskStage.FAILED]
        ]

        # 3. 稳定排序 (按开始时间)
        visible_tasks.sort(key=lambda x: x.start_time)

        # 4. 填入真实数据 (最多填 max_observed_tasks 个)
        num_to_fill = min(len(visible_tasks), self.max_observed_tasks)

        for idx in range(num_to_fill):
            task = visible_tasks[idx]
            is_visible = 0.0  # 掩码位

            # 默认目标位置为 0 (表示未知)
            t_pos_x, t_pos_y, t_pos_z = 0.0, 0.0, 0.0

            # PENDING 状态下，坐标虽然存在于上帝视角，但在观测里必须屏蔽！
            if task.stage != TaskStage.PENDING:
                target_node = next((n for n in self.nodes if n.node_id == task.target_node_id), None)
                if target_node:
                    is_visible = 1.0  # 标记为可见
                    t_pos_x = target_node.pos[0] / self.cfg.space_size_x
                    t_pos_y = target_node.pos[1] / self.cfg.space_size_y
                    t_pos_z = target_node.pos[2] / self.cfg.space_size_z

            start_idx = idx * self.task_feat_dim
            end_idx = start_idx + self.task_feat_dim

            # 构建特征 [剩余时间, 阶段, 可见性, X, Y, Z]
            feat = np.array([
                task.time_window / self.cfg.task_duration,
                task.stage.value / 4.0,  # 归一化阶段
                is_visible,
                t_pos_x,
                t_pos_y,
                t_pos_z
            ], dtype=np.float32)

            task_state_flat[start_idx: end_idx] = feat

        # 注意：不需要写 else 分支补零，因为 task_state_flat 初始化时就是全零的。
        # 如果 num_to_fill < max_observed_tasks，剩下的位置自然保持为 0，即 Padding。

        # ==================== 组装每个 Agent 的 Obs ====================

        for i in range(self.n_agents):
            agent = self.nodes[i]

            # A. 自身状态 (10维) - 注意这里 obs_dim 定义是 10
            # (如果你之前的代码里改过维度，请确保这里和 obs_dim 对应)
            has_msg = 1.0 if len(agent.tx_buffer) > 0 else 0.0

            self_state = np.array([
                agent.energy / 100.0,
                agent.load,
                has_msg,
                agent.raw_capability[0] / 10.0,
                agent.raw_capability[1] / 10.0,
                agent.raw_capability[2] / 10.0,
                agent.raw_capability[3] / 10.0,
                agent.pos[0] / self.cfg.space_size_x,
                agent.pos[1] / self.cfg.space_size_y,
                agent.pos[2] / self.cfg.space_size_z
            ], dtype=np.float32)

            # B. 罗盘特征 (Compass Feat)
            compass_feat = np.zeros(self.compass_dim, dtype=np.float32)

            # --- 1. 寻找最近的关键队友 (Mate) ---
            target_mate_type = None
            if agent.node_type == NodeType.DECISION:
                target_mate_type = NodeType.INTERCEPTOR
            elif agent.node_type == NodeType.INTERCEPTOR:
                target_mate_type = NodeType.DECISION

            if target_mate_type:
                best_dist = float('inf')
                best_mate_idx = -1

                for idx, node in enumerate(self.nodes):
                    if node.node_type == target_mate_type and node.alive:
                        d = dist_matrix[i, idx]
                        if d < best_dist:
                            best_dist = d
                            best_mate_idx = idx

                if best_mate_idx != -1:
                    compass_feat[0] = diff_matrix_norm[i, best_mate_idx, 0]
                    compass_feat[1] = diff_matrix_norm[i, best_mate_idx, 1]
                    compass_feat[2] = diff_matrix_norm[i, best_mate_idx, 2]

            # --- 2. 寻找最近的"已知"目标 (Target) ---
            if agent.node_type == NodeType.INTERCEPTOR:
                best_t_dist = float('inf')
                best_t_idx = -1
                known_target_ids = set()

                # S 节点感知
                if agent.node_type == NodeType.SENSOR:
                    det_range, _, _ = agent.get_actual_ranges(self.cfg)
                    for t_node in self.target_nodes_list:
                        if np.linalg.norm(agent.pos - t_node.pos) <= det_range:
                            known_target_ids.add(t_node.node_id)

                # D/I 节点通过 Buffer 感知
                for item in agent.data_buffer:
                    if isinstance(item, Message) and 'target_id' in item.content:
                        known_target_ids.add(item.content['target_id'])
                    elif isinstance(item, dict) and 'task_id' in item:
                        # D节点的情报里只有 task_id，需要反查 target_id (如果需要的话)
                        pass

                for idx, node in enumerate(self.nodes):
                    if node.node_type == NodeType.TARGET and node.alive:
                        if node.node_id in known_target_ids:
                            d = dist_matrix[i, idx]
                            if d < best_t_dist:
                                best_t_dist = d
                                best_t_idx = idx

                if best_t_idx != -1:
                    compass_feat[3] = diff_matrix_norm[i, best_t_idx, 0]
                    compass_feat[4] = diff_matrix_norm[i, best_t_idx, 1]
                    compass_feat[5] = diff_matrix_norm[i, best_t_idx, 2]

            # C. Buffer 特征 (7维)
            buffer_feat = np.zeros(self.buffer_feat_dim, dtype=np.float32)
            target_msg = None
            if len(agent.data_buffer) > 0:
                item = agent.data_buffer[0]
                if isinstance(item, Message):
                    target_msg = item

            if target_msg is None and len(agent.tx_buffer) > 0:
                target_msg = agent.tx_buffer[0]

            if target_msg is not None and isinstance(target_msg, Message):
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
                    if t_node:
                        rel_vec = (t_node.pos - agent.pos)
                        buffer_feat[4] = rel_vec[0] / self.cfg.space_size_x
                        buffer_feat[5] = rel_vec[1] / self.cfg.space_size_y
                        buffer_feat[6] = rel_vec[2] / self.cfg.space_size_z

            # D. 邻居特征 (K * 7)
            dists = dist_matrix[i].copy()
            dists[i] = np.inf  # 排除自己
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

            # E. 拼接最终 Obs
            # 顺序: [自身10, Buffer7, Compass6, Task(Fixed_Padding), Neighbor]
            obs = np.concatenate([self_state, buffer_feat, compass_feat, task_state_flat, neighbor_state])
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
        构建紧凑的 Global State (上帝视角)
        内容: [所有智能体 Pos(3)+Type(1)+Alive(1), 所有任务 Pos(3)+Stage(1)]
        """
        # 1. 智能体信息 (N_agents * 5)
        agent_feats = []
        # A. 填入真实存在的智能体 (n_agents 个)
        for n in self.nodes[:self.n_agents]:
            feat = [
                n.pos[0] / self.cfg.space_size_x,  # [修改]
                n.pos[1] / self.cfg.space_size_y,  # [修改]
                n.pos[2] / self.cfg.space_size_z,  # [修改]
                n.node_type.value / 3.0,
                1.0 if n.alive else 0.0
            ]
            agent_feats.extend(feat)

        # B. [核心] 填充剩余的空位 (Limit - n_agents 个)
        padding_count = self.max_agents_limit - self.n_agents
        if padding_count > 0:
            # 补全 0.0
            agent_feats.extend([0.0] * (5 * padding_count))

        # 2. 任务/目标信息 (N_targets * 4)
        task_feats = []
        # A. 填入真实存在的任务 (n_target 个)
        sorted_tasks = sorted(self.tasks, key=lambda t: t.task_id)  # 排序确保顺序稳定
        # 即使任务不可见，上帝视角也是知道目标位置的
        for t in self.tasks:
            target_node = self.nodes[t.target_node_id]
            t_feat = [
                target_node.pos[0] / self.cfg.space_size_x,  # [修改]
                target_node.pos[1] / self.cfg.space_size_y,  # [修改]
                target_node.pos[2] / self.cfg.space_size_z,  # [修改]
                t.stage.value / 4.0
            ]
            task_feats.extend(t_feat)

        # B. [核心] 填充剩余的任务空位
        padding_tasks = self.max_targets_limit - len(sorted_tasks)
        if padding_tasks > 0:
            task_feats.extend([0.0] * (4 * padding_tasks))
        # 拼接
        global_state = np.array(agent_feats + task_feats, dtype=np.float32)

        # 返回 N 份 (MAPPO 格式要求)
        return [global_state.copy() for _ in range(self.n_agents)]

    def get_available_actions(self):
        """
        根据节点类型和状态，屏蔽无效动作
        Action Structure: [Strategy(3), Candidate(5), Movement(7)]
        """
        avail_actions = []
        for i in range(self.n_agents):
            agent = self.nodes[i]
            # 初始化为全 0 (全部不可用)
            mask = np.zeros(3 + self.max_candidates + 7, dtype=np.float32)

            # --- 1. Strategy Mask (0:NoOp, 1:Produce, 2:Comm) ---
            mask[0] = 1.0  # NoOp 永远可用

            # 只有特定条件下才能执行 Produce (Detect/Command/Strike)
            if agent.node_type == NodeType.SENSOR:
                mask[1] = 1.0  # S 总是可以尝试探测
            elif agent.node_type == NodeType.DECISION:
                # 只有 Buffer 有情报才能决策
                if len(agent.data_buffer) > 0:
                    mask[1] = 1.0
            elif agent.node_type == NodeType.INTERCEPTOR:
                # 只有 Buffer 有指令才能打击
                if len(agent.data_buffer) > 0:
                    mask[1] = 1.0

            # 只有 Tx Buffer 有消息才能通信
            if len(agent.tx_buffer) > 0:
                mask[2] = 1.0

            # --- 2. Candidate Mask (通信对象) ---
            # 计算有效邻居数量 (需要复用 get_obs 里的逻辑，或者简单判断存活)
            # 这里简化处理：假设前 N-1 个 Candidate 对应真实的邻居
            # 更严谨的做法是在 Env 里维护一个 neighbor_count
            # 暂时开启所有 Candidate，因为如果选错在 step 里会 pass，影响稍小
            # 但最好是计算一下距离内的邻居数
            # (为简化代码，此处暂时置 1，建议后续优化)
            mask[3: 3 + self.max_candidates] = 1.0

            # --- 3. Movement Mask (0~6) ---
            # 简单版：全部允许。进阶版：判断边界，如果在边缘则屏蔽对应方向
            mask[3 + self.max_candidates:] = 1.0

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