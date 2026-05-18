"""
环境封装器 - 适配MAPPO算法的接口
[修复版] 添加了自动reset逻辑
"""
import numpy as np
from typing import List, Tuple, Dict
# 引入通用的、真正的多进程 Wrapper
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

class ShareSubprocVecEnv:
    """
    用于MAPPO的向量化环境包装器
    支持共享观测和子进程并行
    """
    def __init__(self, env_fns):
        """
        Args:
            env_fns: 环境创建函数列表
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

        env = self.envs[0]
        self.n_agents = env.n_agents
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space

    def step(self, actions: np.ndarray) -> Tuple:
        results = []
        for i in range(self.num_envs):
            # 获取环境返回的 6 个值
            obs, s_obs, rew, done, info, avail = self.envs[i].step(actions[i])
            if np.any(done):
                # 自动重置，获取新回合的 3 个值
                obs, s_obs, avail = self.envs[i].reset()
            results.append((obs, s_obs, rew, done, info, avail))

        # 使用 zip(*results) 优雅地解包并转换成 numpy 数组
        obs, s_obs, rews, dones, infos, avails = zip(*results)
        return (np.array(obs), np.array(s_obs),
                np.array(rews).reshape(self.num_envs, self.n_agents, 1),
                np.array(dones), list(infos), np.array(avails))

    def reset(self) -> Tuple:
        results = [env.reset() for env in self.envs]
        # 解包环境返回的 (obs, share_obs, avail_acts)
        obs, s_obs, avails = zip(*results)
        return np.array(obs), np.array(s_obs), np.array(avails)

    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()

    def seed(self, seeds: List[int]):
        """设置随机种子"""
        for i, env in enumerate(self.envs):
            env.seed(seeds[i])


class DummyVecEnv:
    """
    简单的向量化环境包装器（单进程）
    [修复版] 添加了自动reset逻辑
    """
    def __init__(self, env_fns):
        """
        Args:
            env_fns: 环境创建函数列表
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

        env = self.envs[0]
        self.n_agents = env.n_agents
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space

    def step(self, actions: np.ndarray) -> Tuple:
        """执行动作，返回 (obs, share_obs, rewards, dones, infos, avail_acts)"""
        results = []

        for i, env in enumerate(self.envs):
            # 1. 调用环境 step，解包 6 个返回值
            obs, s_obs, rew, done, info, avail = env.step(actions[i])

            # 2. 如果环境结束，执行自动重置
            if np.any(done):
                # 注意：env.reset() 现在返回 3 个值
                obs, s_obs, avail = env.reset()

            results.append((obs, s_obs, rew, done, info, avail))

        # 3. 统一解包所有环境的结果
        obs, s_obs, rews, dones, infos, avails = zip(*results)

        # 4. 转换为符合 MAPPO Runner 要求的格式
        return (np.array(obs),
                np.array(s_obs),
                np.array(rews).reshape(self.num_envs, self.n_agents, 1),
                np.array(dones),
                list(infos),
                np.array(avails))

    def reset(self) -> Tuple:
        """重置所有环境，返回 (obs, share_obs, avail_acts)"""
        # 每个 env.reset() 现在返回 (obs, s_obs, avail)
        results = [env.reset() for env in self.envs]

        # 使用 zip(*results) 将结果解包为三个独立的列表
        obs, share_obs, avail_acts = zip(*results)

        # 转换为 numpy 数组并返回
        return np.array(obs), np.array(share_obs), np.array(avail_acts)

    def close(self):
        """关闭"""
        for env in self.envs:
            env.close()

    def seed(self, seeds: List[int]):
        """设置种子"""
        for i, env in enumerate(self.envs):
            env.seed(seeds[i])


def make_train_env(env_name: str, seed: int, n_threads: int, args) -> DummyVecEnv:
    """
    创建训练环境
    """
    def get_env_fn(rank):
        def init_env():
            from .KillChain_Env import KillChainEnv
            env = KillChainEnv(args, rank=rank, eval_mode=False)
            env.seed(seed + rank * 1000)
            return env
        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])

def make_eval_env(env_name: str, seed: int, n_threads: int, args) -> DummyVecEnv:
    """
    创建评估环境
    """
    def get_env_fn(rank):
        def init_env():
            if env_name == "KillChain":
                from .KillChain_Env import KillChainEnv
                env = KillChainEnv(args, rank=rank, eval_mode=True)
            else:
                raise NotImplementedError(f"Environment {env_name} not supported")

            env.seed(seed * 50000 + rank * 1000)
            return env
        return init_env

    env_fns = [get_env_fn(i) for i in range(n_threads)]
    return DummyVecEnv(env_fns)