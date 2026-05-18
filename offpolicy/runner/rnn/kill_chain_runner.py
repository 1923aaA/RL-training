import numpy as np
import torch
import time
from offpolicy.runner.rnn.base_runner import RecRunner


def _t2n(x):
    """Utility to convert torch tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class KillChainRunner(RecRunner):
    """
    Runner class for the KillChain environment (Off-Policy version).
    """

    def __init__(self, config):
        super(KillChainRunner, self).__init__(config)
        self.env_infos = {
            # --- 基础奖励与状态 ---
            "kill_chain/episode_rewards": [],
            "kill_chain/cooperation_count": [],
            "kill_chain/completed_tasks": [],
            "kill_chain/success_rate": [],
            "kill_chain/is_success": [],

            # --- 杀伤链效率指标 ---
            "kill_chain/avg_pk": [],
            "kill_chain/avg_time_eff": [],
            "kill_chain/paper_temporal_efficiency": [],

            # --- 分角色奖励 ---
            "kill_chain/reward_S": [],
            "kill_chain/reward_D": [],
            "kill_chain/reward_I": [],

            # --- 网络与性能 Metrics ---
            "metrics/avg_hops": [],
            "metrics/avg_ttc": [],
            "metrics/energy_per_kill": [],
            "metrics/avg_buffer_load": [],
            "metrics/msg_throughput": []
        }
        self.num_envs = self.args.n_rollout_threads
        self.episode_length = self.args.episode_length
        self.last_log_time = time.time()
        self.last_log_T = 0

        # [Warmup]
        self.collect_rollout(explore=True, training_episode=False, warmup=True)

    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        num_envs = self.num_envs
        ep_len = self.episode_length

        # 1. 初始化存储字典
        obs_dict, share_obs_dict, avail_acts_dict = {}, {}, {}
        for p_id in self.policy_ids:
            p_agents_num = len(self.trainer.policy_agents[p_id])
            act_dim = self.policy_act_dim[p_id]

            obs_dict[p_id] = np.zeros((ep_len + 1, num_envs, p_agents_num, self.policy_obs_dim[p_id]), dtype=np.float32)
            share_obs_dict[p_id] = np.zeros((ep_len + 1, num_envs, p_agents_num, self.policy_central_obs_dim[p_id]),
                                            dtype=np.float32)
            avail_acts_dict[p_id] = np.zeros((ep_len + 1, num_envs, p_agents_num, act_dim), dtype=np.float32)

        # 2. 环境重置 (Reset)
        obs, share_obs, avail_acts = self.env.reset()

        if avail_acts is None:
            p0_act_dim = self.policy_act_dim[self.policy_ids[0]]
            avail_acts = np.ones((num_envs, self.num_agents, p0_act_dim), dtype=np.float32)

        # RNN 状态初始化
        rnn_states_dict = {}
        for p_id in self.policy_ids:
            idx = self.trainer.policy_agents[p_id]
            obs_dict[p_id][0] = obs[:, idx]
            share_obs_dict[p_id][0] = share_obs[:, idx]
            avail_acts_dict[p_id][0] = avail_acts[:, idx]
            rnn_states_dict[p_id] = self.policies[p_id].init_hidden(len(idx), num_envs)

        # 临时 Buffer
        temp_acts = {}
        temp_rews = {}
        temp_dones = {}
        temp_dones_env = {}

        for p_id in self.policy_ids:
            p_agents_num = len(self.trainer.policy_agents[p_id])
            act_dim = self.policy_act_dim[p_id]

            temp_acts[p_id] = np.zeros((ep_len, num_envs, p_agents_num, act_dim), dtype=np.float32)
            temp_rews[p_id] = np.zeros((ep_len, num_envs, p_agents_num, 1), dtype=np.float32)
            temp_dones[p_id] = np.zeros((ep_len, num_envs, p_agents_num, 1), dtype=np.float32)
            temp_dones_env[p_id] = np.zeros((ep_len, num_envs, 1), dtype=np.float32)

        init_rnn_states = {}
        for p_id in self.policy_ids:
            n_agents = len(self.trainer.policy_agents[p_id])
            init_rnn_states[p_id] = _t2n(rnn_states_dict[p_id]).reshape(num_envs, n_agents, -1).copy()

        # 3. 步进循环 (Step Loop)
        for step in range(ep_len):
            self.trainer.prep_rollout()

            for p_id in self.policy_ids:
                act_dim = self.policy_act_dim[p_id]
                prev_acts = temp_acts[p_id][step - 1] if step > 0 else np.zeros_like(temp_acts[p_id][0])

                act_ret = self.policies[p_id].get_actions(
                    obs_dict[p_id][step].reshape(-1, self.policy_obs_dim[p_id]),
                    prev_acts.reshape(-1, act_dim),
                    rnn_states_dict[p_id].reshape(-1, self.args.hidden_size),
                    available_actions=avail_acts_dict[p_id][step].reshape(-1, act_dim),
                    t_env=self.total_env_steps,
                    explore=explore
                )

                if len(act_ret) == 2:
                    actions, new_rnn = act_ret
                else:
                    actions, new_rnn = act_ret[0], act_ret[1]

                temp_acts[p_id][step] = _t2n(actions).reshape(num_envs, -1, act_dim)
                rnn_states_dict[p_id] = new_rnn.reshape(-1, num_envs, self.args.hidden_size)

            # 拼装 Environment 所需的动作格式
            combined_actions = np.zeros((num_envs, self.num_agents, 3))

            # [修复核心] 动态获取动作空间切分维度
            try:
                # 兼容 KillChainEnv 的配置 (3, 5, 7)
                act_space = self.env.action_space[0]
                if hasattr(act_space, "nvec"):
                    split_dims = act_space.nvec.tolist()
                elif hasattr(act_space, "high"):
                    split_dims = (act_space.high - act_space.low + 1).tolist()
                else:
                    split_dims = [3, 5, 7]  # 默认 fallback
            except:
                split_dims = [3, 5, 7]

            for p_id in self.policy_ids:
                p_idx = self.trainer.policy_agents[p_id]
                one_hot_acts = temp_acts[p_id][step]

                start_idx = 0
                parsed_acts = []
                for dim in split_dims:
                    dim_size = int(dim)
                    end_idx = start_idx + dim_size
                    action_idx = np.argmax(one_hot_acts[..., start_idx:end_idx], axis=-1)
                    parsed_acts.append(action_idx)
                    start_idx = end_idx

                combined_actions[:, p_idx] = np.stack(parsed_acts, axis=-1)

            # Environment Step
            next_obs, next_share_obs, rewards, dones, infos, next_avail = self.env.step(combined_actions)

            if next_avail is None:
                p0_act_dim = self.policy_act_dim[self.policy_ids[0]]
                next_avail = np.ones((num_envs, self.num_agents, p0_act_dim), dtype=np.float32)

            for p_id in self.policy_ids:
                idx = self.trainer.policy_agents[p_id]
                obs_dict[p_id][step + 1] = next_obs[:, idx]
                share_obs_dict[p_id][step + 1] = next_share_obs[:, idx]
                avail_acts_dict[p_id][step + 1] = next_avail[:, idx]
                temp_rews[p_id][step] = rewards[:, idx].reshape(num_envs, -1, 1)
                temp_dones[p_id][step] = dones[:, idx].reshape(num_envs, -1, 1)
                temp_dones_env[p_id][step] = np.all(dones, axis=1, keepdims=True)

            for i in range(num_envs):
                if np.any(dones[i]) and infos[i]:
                    if len(infos[i]) > 0:
                        first_info = infos[i][0]
                        self._process_env_info(first_info)

        # 4. 插入 Buffer
        if training_episode or warmup:
            self.total_env_steps += ep_len * num_envs
            self.num_episodes_collected += num_envs
            self.buffer.insert(
                num_envs,
                obs_dict,
                share_obs_dict,
                temp_acts,
                temp_rews,
                temp_dones,
                temp_dones_env,
                avail_acts_dict,
                rnn_states=init_rnn_states
            )

        return self.env_infos

    def run(self):
        self.collect_rollout(explore=True, training_episode=True, warmup=False)
        if self.total_env_steps >= self.batch_size * self.episode_length:
            for _ in range(getattr(self.args, "num_updates_per_train", 1)):
                self.train()
        if (self.total_env_steps - self.last_save_T) / self.args.save_interval >= 1:
            self.saver()
            self.last_save_T = self.total_env_steps
            print(f"Model saved at step {self.total_env_steps}")
        if self.total_env_steps % self.args.log_interval == 0:
            self.log()
        return self.total_env_steps

    def _process_env_info(self, final_info):
        for k in self.env_infos.keys():
            if k in final_info: self.env_infos[k].append(final_info[k])
        if "kill_chain/total_reward" in final_info:
            self.env_infos["kill_chain/episode_rewards"].append(final_info["kill_chain/total_reward"])

    def log(self):
        elapsed = time.time() - self.last_log_time
        if elapsed <= 0: elapsed = 1e-5
        fps = int((self.total_env_steps - self.last_log_T) / elapsed)
        self.last_log_time, self.last_log_T = time.time(), self.total_env_steps

        def get_m(k):
            vals = self.env_infos.get(k, [])
            return np.mean(vals) if len(vals) > 0 else 0.0

        # 获取数据
        ep_r = get_m("kill_chain/episode_rewards")
        sr = get_m("kill_chain/success_rate")
        s_avg = get_m("kill_chain/reward_S")
        d_avg = get_m("kill_chain/reward_D")
        i_avg = get_m("kill_chain/reward_I")
        pk = get_m("kill_chain/avg_pk")
        time_eff = get_m("kill_chain/avg_time_eff")
        paper_eff = get_m("kill_chain/paper_temporal_efficiency")

        hops = get_m("metrics/avg_hops")
        ttc = get_m("metrics/avg_ttc")
        energy = get_m("metrics/energy_per_kill")
        load = get_m("metrics/avg_buffer_load")
        throughput = get_m("metrics/msg_throughput")

        coops = get_m("kill_chain/cooperation_count")
        completed = get_m("kill_chain/completed_tasks")

        step_r = ep_r / self.episode_length if self.episode_length > 0 else 0.0

        # 按要求格式打印
        print(
            f"\n[Shared] KillChain Exp {self.args.experiment_name}    {self.num_episodes_collected} episodes, Steps {self.total_env_steps}/{self.num_env_steps}, FPS {fps}.")
        print(
            f"Stats: Ep Reward: {ep_r:.1f}, Success Rate: {sr:.2f}, S_Avg: {s_avg:.1f}, D_Avg: {d_avg:.1f}, I_Avg: {i_avg:.1f},Pk: {pk:.3f}, TimeEff: {time_eff:.3f}，TimeEff(Paper): {paper_eff:.4f}")
        print(
            f"Metrics: Hops: {hops:.4f}, TTC: {ttc:.4f}, Energy/Kill: {energy:.1f}, Load: {load:.4f}, Throughput: {throughput:.4f}")
        print(
            f"Stats: Step Reward: {step_r:.3f}, Ep Reward: {ep_r:.1f}, Co-ops: {coops:.2f}, Completed: {completed:.2f}, Finished Eps: {self.num_episodes_collected}")

        self.log_env(self.env_infos)
        for k in self.env_infos.keys(): self.env_infos[k] = []