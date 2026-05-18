"""
AFSIM KillChain 环境的 VecEnv 包装器。

与原 kill_chain/env_wrappers.py 的区别:
- AFSIM 是单实例仿真, n_rollout_threads 必须 = 1, 不支持子进程并行
- 所以只提供 ShareDummyVecEnv(单环境包装), 不提供 ShareSubprocVecEnv
- step() 的 auto-reset 语义和原版一致
"""
import numpy as np
from typing import List, Tuple


class ShareDummyVecEnv:
    """
    单进程向量环境 (n_envs=1 的 DummyVecEnv).
    接口和 MAPPO 原版完全一致, 方便 runner 直接用。
    """

    def __init__(self, env_fns):
        assert len(env_fns) == 1, "AFSIM 场景不支持多实例并行, n_rollout_threads 必须 = 1"
        self.envs = [fn() for fn in env_fns]
        self.num_envs = 1

        env = self.envs[0]
        self.n_agents = env.n_agents
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space

    def step(self, actions: np.ndarray) -> Tuple:
        """
        执行动作, 返回 (obs, share_obs, rewards, dones, infos, avail_acts)
        自带 auto-reset: 任一 agent done 就 reset 整个环境。

        Args:
            actions: shape (num_envs=1, n_agents, action_dim)
        Returns:
            各字段第 0 维都是 num_envs (此处 =1)
        """
        results = []
        for i, env in enumerate(self.envs):
            obs, s_obs, rew, done, info, avail = env.step(actions[i])

            if np.any(done):
                # auto-reset (reset 返回 3 值)
                obs, s_obs, avail = env.reset()

            results.append((obs, s_obs, rew, done, info, avail))

        obs, s_obs, rews, dones, infos, avails = zip(*results)

        return (
            np.array(obs),
            np.array(s_obs),
            np.array(rews).reshape(self.num_envs, self.n_agents, 1),
            np.array(dones),
            list(infos),
            np.array(avails),
        )

    def reset(self) -> Tuple:
        """返回 (obs, share_obs, avail_acts), 每个 shape 第 0 维 = num_envs"""
        results = [env.reset() for env in self.envs]
        obs, share_obs, avail_acts = zip(*results)
        return np.array(obs), np.array(share_obs), np.array(avail_acts)

    def close(self):
        for env in self.envs:
            env.close()

    def seed(self, seeds: List[int]):
        for i, env in enumerate(self.envs):
            env.seed(seeds[i])


def make_train_env(env_name: str, seed: int, n_threads: int, args):
    """
    创建训练环境。AFSIM 只支持 n_threads=1。
    """
    if n_threads != 1:
        raise ValueError(
            f"AFSIM KillChain 不支持并行仿真, n_rollout_threads 必须 = 1 (当前 {n_threads})"
        )

    def get_env_fn(rank):
        def init_env():
            from onpolicy.envs.afsim_kill_chain.AFSIMKillChainEnv import AFSIMKillChainEnv
            env = AFSIMKillChainEnv(args, rank=rank, eval_mode=False)
            env.seed(seed + rank * 1000)
            return env
        return init_env

    return ShareDummyVecEnv([get_env_fn(0)])


def make_eval_env(env_name: str, seed: int, n_threads: int, args):
    """创建评估环境"""
    if n_threads != 1:
        raise ValueError(
            f"AFSIM KillChain 不支持并行仿真, n_eval_rollout_threads 必须 = 1 (当前 {n_threads})"
        )

    def get_env_fn(rank):
        def init_env():
            from onpolicy.envs.afsim_kill_chain.AFSIMKillChainEnv import AFSIMKillChainEnv
            env = AFSIMKillChainEnv(args, rank=rank, eval_mode=True)
            env.seed(seed * 50000 + rank * 1000)
            return env
        return init_env

    return ShareDummyVecEnv([get_env_fn(0)])