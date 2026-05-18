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

        self.train_episode_rewards = []
        self.train_episode_coops = []
        self.train_episode_completed = []
        self.train_episode_pks = []
        self.train_episode_time_effs = []
        self.train_episode_paper_effs = []

        self.train_episode_success_rates = []
        self.train_episode_completion_ratios = []
        self.train_episode_clear_rates = []
        self.train_episode_reward_per_target = []

        self.train_episode_reward_S = []
        self.train_episode_reward_D = []
        self.train_episode_reward_I = []

        self.train_episode_avg_hops = []
        self.train_episode_avg_ttc = []
        self.train_episode_energy_per_kill = []
        self.train_episode_avg_buffer_load = []
        self.train_episode_msg_throughput = []

        # 按目标数分桶
        self.bucket_completion = {}
        self.bucket_clear = {}
        self.bucket_reward_per_target = {}

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                if not self.all_args.use_centralized_V:
                    share_obs = obs

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic
                self.insert(data)

                for t in range(self.n_rollout_threads):
                    if np.any(dones[t]):
                        final_info = infos[t][0]

                        if "kill_chain/reward_S" in final_info:
                            self.train_episode_reward_S.append(final_info["kill_chain/reward_S"])
                        if "kill_chain/reward_D" in final_info:
                            self.train_episode_reward_D.append(final_info["kill_chain/reward_D"])
                        if "kill_chain/reward_I" in final_info:
                            self.train_episode_reward_I.append(final_info["kill_chain/reward_I"])

                        if "kill_chain/success_rate" in final_info:
                            self.train_episode_success_rates.append(final_info["kill_chain/success_rate"])
                        if "kill_chain/total_reward" in final_info:
                            self.train_episode_rewards.append(final_info["kill_chain/total_reward"])
                        if "kill_chain/cooperation_count" in final_info:
                            self.train_episode_coops.append(final_info["kill_chain/cooperation_count"])
                        if "kill_chain/completed_tasks" in final_info:
                            self.train_episode_completed.append(final_info["kill_chain/completed_tasks"])
                        if "kill_chain/avg_pk" in final_info:
                            self.train_episode_pks.append(final_info["kill_chain/avg_pk"])
                        if "kill_chain/avg_time_eff" in final_info:
                            self.train_episode_time_effs.append(final_info["kill_chain/avg_time_eff"])
                        if "kill_chain/paper_temporal_efficiency" in final_info:
                            self.train_episode_paper_effs.append(final_info["kill_chain/paper_temporal_efficiency"])

                        if "kill_chain/completion_ratio" in final_info:
                            self.train_episode_completion_ratios.append(final_info["kill_chain/completion_ratio"])
                        if "kill_chain/clear_rate" in final_info:
                            self.train_episode_clear_rates.append(final_info["kill_chain/clear_rate"])
                        if "kill_chain/reward_per_target" in final_info:
                            self.train_episode_reward_per_target.append(final_info["kill_chain/reward_per_target"])

                        if "metrics/avg_hops" in final_info:
                            self.train_episode_avg_hops.append(final_info["metrics/avg_hops"])
                        if "metrics/avg_ttc" in final_info:
                            self.train_episode_avg_ttc.append(final_info["metrics/avg_ttc"])
                        if "metrics/energy_per_kill" in final_info:
                            self.train_episode_energy_per_kill.append(final_info["metrics/energy_per_kill"])
                        if "metrics/avg_buffer_load" in final_info:
                            self.train_episode_avg_buffer_load.append(final_info["metrics/avg_buffer_load"])
                        if "metrics/msg_throughput" in final_info:
                            self.train_episode_msg_throughput.append(final_info["metrics/msg_throughput"])

                        # 分桶统计
                        if "kill_chain/current_n_target" in final_info:
                            n_t = int(final_info["kill_chain/current_n_target"])

                            if "kill_chain/completion_ratio" in final_info:
                                self.bucket_completion.setdefault(n_t, []).append(
                                    final_info["kill_chain/completion_ratio"])
                            if "kill_chain/clear_rate" in final_info:
                                self.bucket_clear.setdefault(n_t, []).append(final_info["kill_chain/clear_rate"])
                            if "kill_chain/reward_per_target" in final_info:
                                self.bucket_reward_per_target.setdefault(n_t, []).append(
                                    final_info["kill_chain/reward_per_target"])

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()

                avg_step_reward = np.mean(self.buffer.rewards)
                n_finished = len(self.train_episode_rewards)

                log_S = np.mean(self.train_episode_reward_S) if self.train_episode_reward_S else 0.0
                log_D = np.mean(self.train_episode_reward_D) if self.train_episode_reward_D else 0.0
                log_I = np.mean(self.train_episode_reward_I) if self.train_episode_reward_I else 0.0

                valid_hops = [h for h in self.train_episode_avg_hops if h > 0.001]
                log_hops = np.mean(valid_hops) if valid_hops else 0.0

                valid_ttc = [t for t in self.train_episode_avg_ttc if t > 0.001]
                log_ttc = np.mean(valid_ttc) if valid_ttc else 0.0

                log_energy = np.mean(self.train_episode_energy_per_kill) if self.train_episode_energy_per_kill else 0.0
                log_load = np.mean(self.train_episode_avg_buffer_load) if self.train_episode_avg_buffer_load else 0.0
                log_throughput = np.mean(
                    self.train_episode_msg_throughput) if self.train_episode_msg_throughput else 0.0

                log_success_rate = np.mean(
                    self.train_episode_success_rates) if self.train_episode_success_rates else 0.0
                log_completion_ratio = np.mean(
                    self.train_episode_completion_ratios) if self.train_episode_completion_ratios else 0.0
                log_clear_rate = np.mean(self.train_episode_clear_rates) if self.train_episode_clear_rates else 0.0
                log_reward_per_target = np.mean(
                    self.train_episode_reward_per_target) if self.train_episode_reward_per_target else 0.0

                if n_finished > 0:
                    log_reward = np.mean(self.train_episode_rewards)
                    log_coops = np.mean(self.train_episode_coops)
                    log_completed = np.mean(self.train_episode_completed)
                    log_pk = np.mean(self.train_episode_pks) if self.train_episode_pks else 0.0
                    log_time_eff = np.mean(self.train_episode_time_effs) if self.train_episode_time_effs else 0.0
                    log_paper_eff = np.mean(self.train_episode_paper_effs) if self.train_episode_paper_effs else 0.0
                else:
                    log_reward = 0.0
                    log_coops = 0.0
                    log_completed = 0.0
                    log_pk = 0.0
                    log_time_eff = 0.0
                    log_paper_eff = 0.0

                print("\n[Shared] KillChain Exp {}    {} episodes, Steps {}/{}, FPS {}."
                      .format(self.experiment_name, episode, total_num_steps, self.num_env_steps,
                              int(total_num_steps / (end - start))))

                print(
                    f"Stats: Ep Reward: {log_reward:.1f}, Success Rate: {log_success_rate:.2f}, "
                    f"Completion Ratio: {log_completion_ratio:.3f}, Clear Rate: {log_clear_rate:.3f}, "
                    f"Reward/Target: {log_reward_per_target:.3f}, "
                    f"S_Avg: {log_S:.1f}, D_Avg: {log_D:.1f}, I_Avg: {log_I:.1f}, "
                    f"Pk: {log_pk:.3f}, TimeEff: {log_time_eff:.3f}, TimeEff(Paper): {log_paper_eff:.4f}"
                )
                print(
                    f"Metrics: Hops: {log_hops:.4f}, TTC: {log_ttc:.4f}, Energy/Kill: {log_energy:.1f}, "
                    f"Load: {log_load:.4f}, Throughput: {log_throughput:.4f}"
                )
                print(
                    f"Stats: Step Reward: {avg_step_reward:.3f}, "
                    f"Ep Reward: {log_reward:.1f}, Co-ops: {log_coops:.2f}, "
                    f"Completed: {log_completed:.2f}, Finished Eps: {n_finished}"
                )

                # 分桶打印
                if len(self.bucket_completion) > 0:
                    bucket_lines = []
                    for k in sorted(self.bucket_completion.keys()):
                        c = np.mean(self.bucket_completion[k]) if self.bucket_completion[k] else 0.0
                        cr = np.mean(self.bucket_clear.get(k, [])) if self.bucket_clear.get(k, []) else 0.0
                        rpt = np.mean(self.bucket_reward_per_target.get(k, [])) if self.bucket_reward_per_target.get(k,
                                                                                                                     []) else 0.0
                        bucket_lines.append(f"N={k}: comp={c:.3f}, clear={cr:.3f}, rpt={rpt:.3f}")
                        train_infos[f"bucket/{k}/completion_ratio"] = c
                        train_infos[f"bucket/{k}/clear_rate"] = cr
                        train_infos[f"bucket/{k}/reward_per_target"] = rpt
                    print("Buckets -> " + " | ".join(bucket_lines))

                train_infos["kill_chain/success_rate"] = log_success_rate
                train_infos["kill_chain/reward_S"] = log_S
                train_infos["kill_chain/reward_D"] = log_D
                train_infos["kill_chain/reward_I"] = log_I
                train_infos["kill_chain/avg_pk"] = log_pk
                train_infos["kill_chain/avg_time_eff"] = log_time_eff
                train_infos["kill_chain/paper_temporal_efficiency"] = log_paper_eff

                train_infos["kill_chain/completion_ratio"] = log_completion_ratio
                train_infos["kill_chain/clear_rate"] = log_clear_rate
                train_infos["kill_chain/reward_per_target"] = log_reward_per_target

                train_infos["average_step_rewards"] = avg_step_reward
                train_infos["kill_chain/episode_rewards"] = log_reward
                train_infos["kill_chain/cooperation_count"] = log_coops
                train_infos["kill_chain/completed_tasks"] = log_completed

                train_infos["metrics/avg_hops"] = log_hops
                train_infos["metrics/avg_ttc"] = log_ttc
                train_infos["metrics/energy_per_kill"] = log_energy
                train_infos["metrics/avg_buffer_load"] = log_load
                train_infos["metrics/msg_throughput"] = log_throughput

                self.log_train(train_infos, total_num_steps)

                self.train_episode_rewards = []
                self.train_episode_coops = []
                self.train_episode_completed = []
                self.train_episode_pks = []
                self.train_episode_time_effs = []
                self.train_episode_paper_effs = []

                self.train_episode_success_rates = []
                self.train_episode_completion_ratios = []
                self.train_episode_clear_rates = []
                self.train_episode_reward_per_target = []

                self.train_episode_avg_hops = []
                self.train_episode_avg_ttc = []
                self.train_episode_energy_per_kill = []
                self.train_episode_avg_buffer_load = []
                self.train_episode_msg_throughput = []

                self.train_episode_reward_S = []
                self.train_episode_reward_D = []
                self.train_episode_reward_I = []

                self.bucket_completion = {}
                self.bucket_clear = {}
                self.bucket_reward_per_target = {}

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        if not self.all_args.use_centralized_V:
            share_obs = obs
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]))

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

    def save(self):
        """保存模型"""
        # 1. 获取 policy 里的 actor 和 critic 网络
        policy_actor = self.trainer.policy.actor
        policy_critic = self.trainer.policy.critic

        # 2. 确保保存目录存在 (基类通常会创建 self.save_dir)
        # 路径通常是: results/.../runX/models/

        # 3. 保存权重
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

        # [新增] 确保保存 vnorm
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")

        # 如果你想区分步数保存（比如 actor_1000.pt），可以用下面这行代替：
        # torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/actor_{self.total_num_steps}.pt")