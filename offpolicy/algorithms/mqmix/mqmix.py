import torch
import numpy as np
from offpolicy.algorithms.mqmix.algorithm.mQMixPolicy import M_QMixPolicy
from offpolicy.utils.util import get_update_info, check


class M_QMix:
    def __init__(self, args, num_agents, policies, policy_mapping_fn, device=torch.device("cpu"), episode_length=None):
        self.args = args
        self.num_agents = num_agents
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.device = device
        self.episode_length = episode_length
        self.action_space = self.policies[self.policy_mapping_fn(0)].act_space
        self.data_chunk_length = args.data_chunk_length
        self.use_recurrent_policy = args.use_rnn_layer
        self._use_recurrent_policy = args.use_rnn_layer
        self.use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy_agents = {policy_id: [] for policy_id in policies.keys()}
        for agent_id in range(num_agents):
            policy_id = policy_mapping_fn(agent_id)
            if policy_id in self.policy_agents:
                self.policy_agents[policy_id].append(agent_id)
            else:
                self.policy_agents[policy_id] = [agent_id]
        self.mixer = self.policies[self.policy_mapping_fn(0)].mixer

    def train_policy_on_batch(self, batch):
        # 1. 解包 Batch
        if len(batch) == 10:
            obs, share_obs, acts, rewards, dones, dones_env, avail_acts, rnn_states, valid_transition, _ = batch
        elif len(batch) == 9:
            obs, share_obs, acts, rewards, dones, dones_env, avail_acts, rnn_states, valid_transition = batch
        else:
            share_obs, obs, acts, rewards, dones, dones_env, valid_transition, avail_acts, rnn_states, rnn_states_critic = batch

        def _extract(data):
            return list(data.values())[0] if isinstance(data, dict) else data

        obs = _extract(obs)
        share_obs = _extract(share_obs)
        acts = _extract(acts)
        rewards = _extract(rewards)
        dones = _extract(dones)
        dones_env = _extract(dones_env)
        avail_acts = _extract(avail_acts)
        rnn_states = _extract(rnn_states)
        valid_transition = _extract(valid_transition)

        # =============================================================================
        # [针对性适配] 维度修正: 强制转为 (Time, Batch, Agents, Dim)
        # =============================================================================
        def _fix_dims(x):
            t = check(x).to(**self.tpdv)
            shape = t.shape

            # RecReplayBuffer 特有格式: (Agents, Time, Batch, Dim) -> 转置为 (Time, Batch, Agents, Dim)
            if t.dim() == 4 and shape[0] == self.num_agents:
                return t.permute(1, 2, 0, 3)

            # 兼容其他格式 (Batch, Time, Agents, Dim)
            if t.dim() == 4 and shape[0] == self.args.batch_size:
                return t.permute(1, 0, 2, 3)

            # 处理 (Batch, Time, Dim) 或 (Time, Batch, Dim) 的 3D 数据
            # RecReplayBuffer 的 dones_env 通常是 (T, B, 1) 或者 (B, T, 1)
            if t.dim() == 3:
                # 如果是 (B, T, D) 且 B=BatchSize
                if shape[0] == self.args.batch_size:
                    t = t.permute(1, 0, 2)
                # 扩展为 4D: (T, B, 1, 1) 以便广播
                return t.unsqueeze(2)

            return t

        def _fix_shared(x):
            t = check(x).to(**self.tpdv)
            # 处理 share_obs
            if t.dim() == 3:  # (T, B, D) -> (T, B, 1, D)
                if t.shape[0] == self.args.batch_size: t = t.permute(1, 0, 2)
                return t.unsqueeze(2)
            if t.dim() == 4 and t.shape[0] == self.num_agents:  # (N, T, B, D) -> (T, B, 1, D)
                t = t.permute(1, 2, 0, 3)
                return t[:, :, 0:1, :]
            if t.dim() == 4 and t.shape[0] == self.args.batch_size:  # (B, T, N, D)
                t = t.permute(1, 0, 2, 3)
                return t[:, :, 0:1, :]
            return t

        # 应用维度修正
        obs_batch = _fix_dims(obs)  # (L+1, B, N, O)
        act_batch = _fix_dims(acts)  # (L, B, N, A)
        avail_act_batch = _fix_dims(avail_acts)  # (L+1, B, N, A)
        reward_batch = _fix_dims(rewards)  # (L, B, N, 1)
        done_batch = _fix_dims(dones)  # (L, B, N, 1)
        share_obs_batch = _fix_shared(share_obs)  # (L+1, B, 1, S)

        # 处理 Mask
        dones_env_t = check(dones_env).to(**self.tpdv)
        if dones_env_t.dim() == 3 and dones_env_t.shape[0] == self.args.batch_size:
            dones_env_t = dones_env_t.permute(1, 0, 2)  # (B, T, 1) -> (T, B, 1)

        # 统一扩展 Mask 为 4D
        if dones_env_t.dim() == 3:
            mask_batch = 1 - dones_env_t.unsqueeze(2)  # (T, B, 1, 1)
        else:
            mask_batch = 1 - _fix_dims(dones_env)

        if valid_transition is not None:
            vt_batch = _fix_dims(valid_transition)
            min_len = min(mask_batch.shape[0], vt_batch.shape[0])
            mask_batch = mask_batch[:min_len] * vt_batch[:min_len]

        # ---------------------------------------------------
        # 准备数据序列 (使用完整长度 L)
        # ---------------------------------------------------
        # Obs: L+1 -> 取前L个做 curr，后L个做 next
        curr_obs = obs_batch[:-1]
        next_obs = obs_batch[1:]

        # Acts: L -> 全部使用
        curr_acts = act_batch

        # Avail / Share: L+1 -> 也是取前L个和后L个
        curr_avail, next_avail = avail_act_batch[:-1], avail_act_batch[1:]
        curr_share, next_share = share_obs_batch[:-1], share_obs_batch[1:]

        # Mask / Rewards: L -> 全部使用 (不要切片!)
        # mask_batch = mask_batch # 保持原样 (L, B, 1, 1)

        # 构建 Previous Action (用于 Q-Eval 输入)
        # 输入给 t 的 prev_action 应该是 a_{t-1}。对于 t=0，是 0。
        T, B, N, A = curr_acts.shape
        zero_action = torch.zeros((1, B, N, A), device=self.device, dtype=torch.float32)
        # 拼接 0 和 前 L-1 个动作 -> 长度 L
        prev_action_batch = torch.cat([zero_action, curr_acts[:-1]], dim=0)

        # RNN States
        rnn_states_batch = check(rnn_states).to(**self.tpdv) if rnn_states is not None else torch.zeros(B, N,
                                                                                                        self.args.hidden_size).to(
            **self.tpdv)
        # 扁平化处理 (如果 buffer 返回的是 stack 后的)
        if rnn_states_batch.dim() == 3 and rnn_states_batch.shape[0] == self.num_agents:
            rnn_states_batch = rnn_states_batch.permute(1, 0, 2)  # (N, B, H) -> (B, N, H)

        # =============================================================================
        # Forward 计算
        # =============================================================================
        policy = self.policies[self.policy_mapping_fn(0)]

        # 1. 获取当前 Q 值
        # 输入: obs(t), prev_act(t) -> 输出 Q(t)
        q_evals, _ = policy.get_q_values(curr_obs, prev_action_batch, rnn_states_batch, curr_acts)

        # 2. 获取目标 Q 值
        # 输入: obs(t+1), prev_act(t+1) == act(t)
        with torch.no_grad():
            if next_obs.shape[2] != curr_acts.shape[2]:
                if next_obs.shape[2] == 1: next_obs = next_obs.expand(-1, -1, curr_acts.shape[2], -1)

            # [核心修复] 直接拼接，不要切片
            # next_obs 是 L 长度，curr_acts 也是 L 长度，完全对齐
            target_input = torch.cat((next_obs, curr_acts), dim=-1)

            # 注意: 这里使用初始 rnn_states，因为我们是一次性推理整个序列
            q_targets, _ = policy.target_critic(target_input, rnn_states_batch)

        # =============================================================================
        # MQMix MultiDiscrete 聚合逻辑
        # =============================================================================

        # A. Chosen Q: Sum(Q * OneHot)
        chosen_action_qvals = torch.sum(q_evals * curr_acts, dim=-1, keepdim=True)

        # B. Target Max Q: Split -> Max -> Sum
        if hasattr(self, 'act_dims'):
            dims = self.act_dims
        elif hasattr(self.action_space, 'nvec'):
            dims = self.action_space.nvec.tolist()
        else:
            from offpolicy.utils.util import get_dim_from_space
            dims = [int(get_dim_from_space(self.action_space))]

        q_target_splits = torch.split(q_targets, dims, dim=-1)
        avail_splits = torch.split(next_avail, dims, dim=-1)

        target_max_q_list = []
        for q_s, mask_s in zip(q_target_splits, avail_splits):
            q_s = q_s.clone()
            q_s[mask_s == 0] = -1e10
            max_q = q_s.max(dim=-1)[0]
            target_max_q_list.append(max_q)

        target_max_qvals = torch.stack(target_max_q_list, dim=-1).sum(dim=-1).unsqueeze(-1)

        # =============================================================================
        # Mixer & Loss
        # =============================================================================
        # Reshape for Mixer: (T*B, N)
        chosen_q_agents = chosen_action_qvals.reshape(-1, N)
        target_max_q_agents = target_max_qvals.reshape(-1, N)

        curr_share_flat = curr_share.reshape(-1, self.mixer.cent_obs_dim)
        next_share_flat = next_share.reshape(-1, self.mixer.cent_obs_dim)

        # Mixer Forward
        # 输出形状 (T*B, 1) -> Reshape 回 (T, B, 1)
        q_total = policy.mixer(chosen_q_agents, curr_share_flat).reshape(T, B, 1)
        target_q_total = policy.target_mixer(target_max_q_agents, next_share_flat).reshape(T, B, 1)

        # 数据对齐 (无需切片，直接使用完整长度)
        rewards_scalar = reward_batch.squeeze(-1)  # (T, B, 1)
        dones_scalar = done_batch.squeeze(-1)  # (T, B, 1)
        mask_mixer = mask_batch.squeeze(-1)  # (T, B, 1)

        # TD Error
        targets = rewards_scalar + self.args.gamma * (1 - dones_scalar) * target_q_total
        td_error = q_total - targets.detach()
        loss = (td_error ** 2 * mask_mixer).sum() / mask_mixer.sum()

        policy.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters, self.args.max_grad_norm)
        policy.optimizer.step()

        return {'loss': loss.item(), 'grad_norm': grad_norm.item(), 'Q_total': q_total.mean().item()}, None, None

    def prep_training(self):
        for p in self.policies.values(): p.prep_training()

    def prep_rollout(self):
        for p in self.policies.values(): p.prep_rollout()

    def soft_target_updates(self):
        for policy_id in self.policies.keys():
            self.policies[policy_id].soft_target_updates()

    def hard_target_updates(self):
        for policy_id in self.policies.keys():
            self.policies[policy_id].hard_target_updates()