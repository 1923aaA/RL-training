import numpy as np
from typing import List, Tuple, Dict
from gym import spaces
from .KillChain_Env import KillChainEnv


class KillChainOffPolicyWrapper(KillChainEnv):
    """
    [修复版] KillChain环境的Off-Policy包装器

    修复点：
    1. reset() 正确解包 (obs, share_obs, available_actions)，解决 ValueError 连接错误。
    2. step() 正确解包 6 个返回值。
    3. 移除了多余的 share_obs 手动拼接（直接使用 Env 计算好的更高效的 share_obs）。
    4. 移除了错误的动作 One-Hot 转换（Runner 已经传了 Indices，Env 也只要 Indices）。
    """

    def __init__(self, args, rank=0):
        super().__init__(args, rank)

        # 适配 ShareSubprocVecEnv 对 num_agents 的调用
        self.num_agents = self.n_agents

    def step(self, actions):
        """
        Args:
            actions: (n_agents, 3) 的离散索引数组 (由 KillChainRunner 解析后传入)
        """
        # [修改] 移除 One-Hot 转换逻辑
        # 因为 KillChainRunner 已经将 Neural Network 输出的 One-Hot 转换为了 Indices
        # 且 KillChainEnv 内部 (如 _process_movement) 也是按索引读取的

        # 1. 执行环境 Step (正确解包 6 个返回值)
        obs, share_obs, rewards, dones, infos, available_actions = super().step(actions)

        # 2. Reward 维度调整: (n_agents,) -> (n_agents, 1)
        if np.ndim(rewards) == 1:
            rewards = rewards[:, np.newaxis]

        # 3. 构造 available_actions (如果环境返回 None，Off-Policy 算法通常需要全1 Mask)
        if available_actions is None:
            # 获取动作空间维度 [3, 10, 7]
            if hasattr(self.action_space[0], 'nvec'):
                act_dims = self.action_space[0].nvec
                total_dim = sum(act_dims)
            else:
                total_dim = 20  # Fallback

            available_actions = np.ones((self.n_agents, total_dim), dtype=np.float32)

        return obs, share_obs, rewards, dones, infos, available_actions

    def reset(self):
        # [关键修复] 正确解包 3 个返回值
        # 原代码直接 obs = super().reset() 导致 obs 变成了一个 tuple
        # 随后的 np.concatenate(obs) 试图连接 obs_dim(90) 和 share_obs_dim(149) 的数组，导致崩溃
        obs, share_obs, available_actions = super().reset()

        # 如果环境 reset 没返回 mask，手动补全
        if available_actions is None:
            if hasattr(self.action_space[0], 'nvec'):
                act_dims = self.action_space[0].nvec
                total_dim = sum(act_dims)
            else:
                total_dim = 20
            available_actions = np.ones((self.n_agents, total_dim), dtype=np.float32)

        return obs, share_obs, available_actions

    def seed(self, seed=None):
        super().seed(seed)