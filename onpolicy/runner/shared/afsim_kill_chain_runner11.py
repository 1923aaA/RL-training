"""
AFSIM KillChain Runner (基于 shared MAPPO).

和原 kill_chain_runner.py 几乎一致, 主要差别:
1. 导入 AFSIMKillChainEnv (环境差异由环境层本身处理, runner 只需调 step/reset)
2. close() 时要调 env.close() 让 bridge 正确关闭
3. [AFSIM 新增] 在训练日志里打印 Bridge 的 DDS 收发统计
   (platform_added, move_update, track_event, strike_event, fire_weapon, sim_reset)
"""
import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class AFSIMKillChainRunner(Runner):
    """
    AFSIM 版训练循环。
    由于 AFSIM 单实例, n_rollout_threads 必须 = 1, 相应 episode 采样速率会比原 KillChain 慢得多。
    """

    def __init__(self, config):
        super(AFSIMKillChainRunner, self).__init__(config)

        # 训练统计 (和 kill_chain_runner 对齐)
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
                (values, actions, action_log_probs,
                 rnn_states, rnn_states_critic, actions_env) = self.collect(step)

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                if not self.all_args.use_centralized_V:
                    share_obs = obs

                data = (obs, share_obs, rewards, dones, infos, available_actions,
                        values, actions, action_log_probs,
                        rnn_states, rnn_states_critic)
                self.insert(data)

                for t in range(self.n_rollout_threads):
                    if np.any(dones[t]):
                        final_info = infos[t][0] if len(infos[t]) > 0 else {}
                        self._collect_episode_stats(final_info)

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                self._log_episode_stats(train_infos, episode, total_num_steps, start)

    # ===================================================================
    # warmup / collect / insert / save (和 kill_chain_runner 完全一致)
    # ===================================================================
    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        if not self.all_args.use_centralized_V:
            share_obs = obs
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic = \
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
                np.concatenate(self.buffer.available_actions[step]),
            )

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions

    def insert(self, data):
        (obs, share_obs, rewards, dones, infos, available_actions,
         values, actions, action_log_probs, rnn_states, rnn_states_critic) = data

        if len(rewards.shape) == 2:
            rewards = rewards[..., np.newaxis]

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        self.buffer.insert(
            share_obs, obs, rnn_states, rnn_states_critic,
            actions, action_log_probs, values, rewards, masks,
            available_actions=available_actions,
        )

    def save(self):
        policy_actor = self.trainer.policy.actor
        policy_critic = self.trainer.policy.critic
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")

    # ===================================================================
    # 统计 / 日志
    # ===================================================================
    def _collect_episode_stats(self, info: dict):
        def push(key, target_list):
            if key in info:
                target_list.append(info[key])

        push("kill_chain/reward_S", self.train_episode_reward_S)
        push("kill_chain/reward_D", self.train_episode_reward_D)
        push("kill_chain/reward_I", self.train_episode_reward_I)
        push("kill_chain/success_rate", self.train_episode_success_rates)
        push("kill_chain/total_reward", self.train_episode_rewards)
        push("kill_chain/cooperation_count", self.train_episode_coops)
        push("kill_chain/completed_tasks", self.train_episode_completed)
        push("kill_chain/avg_pk", self.train_episode_pks)
        push("kill_chain/avg_time_eff", self.train_episode_time_effs)
        push("kill_chain/paper_temporal_efficiency", self.train_episode_paper_effs)
        push("kill_chain/completion_ratio", self.train_episode_completion_ratios)
        push("kill_chain/clear_rate", self.train_episode_clear_rates)
        push("kill_chain/reward_per_target", self.train_episode_reward_per_target)
        push("metrics/avg_hops", self.train_episode_avg_hops)
        push("metrics/avg_ttc", self.train_episode_avg_ttc)
        push("metrics/energy_per_kill", self.train_episode_energy_per_kill)
        push("metrics/avg_buffer_load", self.train_episode_avg_buffer_load)
        push("metrics/msg_throughput", self.train_episode_msg_throughput)

        if "kill_chain/current_n_target" in info:
            n_t = int(info["kill_chain/current_n_target"])
            if "kill_chain/completion_ratio" in info:
                self.bucket_completion.setdefault(n_t, []).append(info["kill_chain/completion_ratio"])
            if "kill_chain/clear_rate" in info:
                self.bucket_clear.setdefault(n_t, []).append(info["kill_chain/clear_rate"])
            if "kill_chain/reward_per_target" in info:
                self.bucket_reward_per_target.setdefault(n_t, []).append(info["kill_chain/reward_per_target"])

    def _log_episode_stats(self, train_infos: dict, episode: int,
                           total_num_steps: int, start_time: float):
        end = time.time()
        avg_step_reward = np.mean(self.buffer.rewards)

        def mean_or_zero(lst):
            return float(np.mean(lst)) if lst else 0.0

        log_reward = mean_or_zero(self.train_episode_rewards)
        log_coops = mean_or_zero(self.train_episode_coops)
        log_completed = mean_or_zero(self.train_episode_completed)
        log_success_rate = mean_or_zero(self.train_episode_success_rates)
        log_completion_ratio = mean_or_zero(self.train_episode_completion_ratios)
        log_clear_rate = mean_or_zero(self.train_episode_clear_rates)
        log_reward_per_target = mean_or_zero(self.train_episode_reward_per_target)

        log_S = mean_or_zero(self.train_episode_reward_S)
        log_D = mean_or_zero(self.train_episode_reward_D)
        log_I = mean_or_zero(self.train_episode_reward_I)

        log_pk = mean_or_zero(self.train_episode_pks)
        log_time_eff = mean_or_zero(self.train_episode_time_effs)
        log_paper_eff = mean_or_zero(self.train_episode_paper_effs)

        valid_hops = [h for h in self.train_episode_avg_hops if h > 0.001]
        log_hops = mean_or_zero(valid_hops)
        valid_ttc = [t for t in self.train_episode_avg_ttc if t > 0.001]
        log_ttc = mean_or_zero(valid_ttc)

        log_energy = mean_or_zero(self.train_episode_energy_per_kill)
        log_load = mean_or_zero(self.train_episode_avg_buffer_load)
        log_throughput = mean_or_zero(self.train_episode_msg_throughput)

        duration = max(1e-6, end - start_time)

        print(f"\n[AFSIM-Shared] Exp {self.experiment_name}  "
              f"Episode {episode}  Steps {total_num_steps}/{self.num_env_steps}  "
              f"FPS {int(total_num_steps / duration)}")
        print(f"  EpReward: {log_reward:.1f} | Success: {log_success_rate:.2f} | "
              f"CompletionRatio: {log_completion_ratio:.3f} | Clear: {log_clear_rate:.3f}")
        print(f"  S_Avg: {log_S:.1f} | D_Avg: {log_D:.1f} | I_Avg: {log_I:.1f} | "
              f"Pk: {log_pk:.3f} | TimeEff: {log_time_eff:.3f}")
        print(f"  Metrics | Hops: {log_hops:.2f}  TTC: {log_ttc:.2f}  "
              f"Energy/Kill: {log_energy:.1f}  Load: {log_load:.3f}  "
              f"Throughput: {log_throughput:.3f}")
        print(f"  StepReward: {avg_step_reward:.3f}  FinishedEps: {len(self.train_episode_rewards)}")

        # [AFSIM 新增] Bridge DDS 收发统计
        self._log_bridge_stats(train_infos)

        # 分桶打印
        if self.bucket_completion:
            bucket_lines = []
            for k in sorted(self.bucket_completion.keys()):
                c = mean_or_zero(self.bucket_completion[k])
                cr = mean_or_zero(self.bucket_clear.get(k, []))
                rpt = mean_or_zero(self.bucket_reward_per_target.get(k, []))
                bucket_lines.append(f"N={k}:comp={c:.3f},clear={cr:.3f},rpt={rpt:.3f}")
                train_infos[f"bucket/{k}/completion_ratio"] = c
                train_infos[f"bucket/{k}/clear_rate"] = cr
                train_infos[f"bucket/{k}/reward_per_target"] = rpt
            print("  Buckets: " + " | ".join(bucket_lines))

        # 汇总到 tensorboard
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

        # 清空
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
        self.bucket_completion = {}
        self.bucket_clear = {}
        self.bucket_reward_per_target = {}

    # ===================================================================
    # [AFSIM 新增] Bridge 通信统计
    # ===================================================================
    def _log_bridge_stats(self, train_infos: dict):
        """
        打印并记录 Bridge 的 DDS 收发统计. 方便观察:
        - track_event_recv: AFSIM 每 step 该发多少 track, 没发 = 传感器/订阅有问题
        - strike_event_recv: 开火后该收到 FIRED+HIT/MISS+TERMINATED 三条, 没收 = 反馈链断了
        - fire_weapon_sent: Python 侧发过多少 FireWeapon
        - sim_reset_sent: episode 之间的 SimReset 次数
        统计值是累计值, 不清零.
        """
        try:
            env0 = self.envs.envs[0] if hasattr(self.envs, "envs") else None
            if env0 is None or not hasattr(env0, "bridge"):
                return
            stats = env0.bridge.get_stats()

            print(
                f"  Bridge | platform_added={stats.get('platform_added_recv', 0)}  "
                f"move_update={stats.get('move_update_recv', 0)}  "
                f"track_evt={stats.get('track_event_recv', 0)}  "
                f"strike_evt={stats.get('strike_event_recv', 0)}"
            )
            print(
                f"         fire_sent={stats.get('fire_weapon_sent', 0)}  "
                f"ctrl_cmd_sent={stats.get('ctrl_cmd_sent', 0)}  "
                f"sim_reset={stats.get('sim_reset_sent', 0)}  "
                f"parse_err={stats.get('parse_errors', 0)}"
            )

            # 把主要指标也汇总到 tensorboard (便于趋势观察)
            for k in ["platform_added_recv", "move_update_recv",
                      "track_event_recv", "strike_event_recv",
                      "fire_weapon_sent", "ctrl_cmd_sent",
                      "sim_reset_sent", "parse_errors"]:
                train_infos[f"bridge/{k}"] = float(stats.get(k, 0))
        except Exception as e:
            # 日志/统计失败不影响训练
            try:
                print(f"  [Bridge stats] err: {e}")
            except Exception:
                pass