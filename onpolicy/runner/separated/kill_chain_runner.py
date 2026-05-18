import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class KillChainRunner(Runner):
    """
    [完全修复版] 修复了统计逻辑：
    1. 正确管理统计列表的生命周期
    2. 在log之后清空统计列表
    3. 修复了奖励累加问题
    """

    def __init__(self, config):
        super(KillChainRunner, self).__init__(config)

        # [新增] 在整个训练过程中维护统计列表
        self.train_episode_rewards = []
        self.train_episode_coops = []
        self.train_episode_completed = []

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # 1. 采样
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # 2. 执行环境交互
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                # 3. 存储数据
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic
                self.insert(data)

                # [修复] 实时检查Done并提取Info
                for t in range(self.n_rollout_threads):
                    if np.any(dones[t]):
                        # 环境结束时，从info中提取统计数据
                        final_info = infos[t][0]  # 所有智能体共享相同的info

                        if "kill_chain/total_reward" in final_info:
                            self.train_episode_rewards.append(final_info["kill_chain/total_reward"])
                        if "kill_chain/cooperation_count" in final_info:
                            self.train_episode_coops.append(final_info["kill_chain/cooperation_count"])
                        if "kill_chain/completed_tasks" in final_info:
                            self.train_episode_completed.append(final_info["kill_chain/completed_tasks"])

            # 4. 训练
            self.compute()
            train_infos = self.train()

            # 5. 日志与保存
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()

                # [修复] 正确计算统计指标

                # A. 微观指标：每一步的平均奖励
                avg_step_reward = np.mean(self.buffer.rewards)

                # B. 宏观指标：本次log_interval内完成的episode统计
                n_finished = len(self.train_episode_rewards)

                if n_finished > 0:
                    log_reward = np.mean(self.train_episode_rewards)
                    log_coops = np.mean(self.train_episode_coops)
                    log_completed = np.mean(self.train_episode_completed)
                else:
                    log_reward = 0.0
                    log_coops = 0.0
                    log_completed = 0.0

                print("\n[Shared] KillChain Exp {}    {} episodes, Steps {}/{}, FPS {}."
                      .format(self.experiment_name, episode, total_num_steps, self.num_env_steps,
                              int(total_num_steps / (end - start))))

                print(f"Stats: Step Reward: {avg_step_reward:.3f}, "
                      f"Ep Reward: {log_reward:.1f}, "
                      f"Co-ops: {log_coops:.1f}, "
                      f"Completed: {log_completed:.1f}, "
                      f"Finished Eps: {n_finished}")

                train_infos["average_step_rewards"] = avg_step_reward
                train_infos["kill_chain/episode_rewards"] = log_reward
                train_infos["kill_chain/cooperation_count"] = log_coops
                train_infos["kill_chain/completed_tasks"] = log_completed

                self.log_train(train_infos, total_num_steps)

                # [关键修复] 清空统计列表，准备下一个log_interval
                self.train_episode_rewards = []
                self.train_episode_coops = []
                self.train_episode_completed = []

    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        if available_actions is not None:
            self.buffer.available_actions[0] = available_actions.copy()

    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        if len(rewards.shape) == 2:
            rewards = rewards[..., np.newaxis]

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks, available_actions=available_actions)