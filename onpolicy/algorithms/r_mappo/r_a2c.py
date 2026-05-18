import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check


class R_A2C():
    """
    Trainer class for A2C to update policies.
    完全独立的 A2C 实现，去除了 PPO 的 Clip 和 Epoch 循环逻辑。
    """

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        # A2C 不需要 clip_param, ppo_epoch
        # 但我们仍然保留 num_mini_batch 以支持小批次梯度下降（显存优化）
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, return_batch, active_masks_batch):
        """
        A2C 的价值损失计算
        区别：移除了 value_clip 逻辑，直接计算预测值与回报的差异
        """
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss = huber_loss(error_original, self.huber_delta)
        else:
            value_loss = mse_loss(error_original)

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def a2c_update(self, sample, update_actor=True):
        """
        A2C 核心更新逻辑
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample

        # 转换为 Tensor
        adv_targ = check(adv_targ).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # 1. 评估当前动作 (Evaluate Actions)
        # 注意：这里不需要 old_action_log_probs，因为 A2C 不算 Ratio
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)

        # 2. Actor Update (Policy Loss)
        # 公式: Loss = - (log_prob * advantage) - (entropy_coef * entropy)
        if self._use_policy_active_masks:
            policy_action_loss = -(action_log_probs * adv_targ * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -(action_log_probs * adv_targ).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # 3. Critic Update (Value Loss)
        value_loss = self.cal_value_loss(values, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, action_log_probs

    def train(self, buffer, update_actor=True):
        """
        A2C 的训练循环
        区别：移除了 ppo_epoch 循环，通常 A2C 遍历一次数据即可
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        num_updates = 0

        if self._use_recurrent_policy:
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
        else:
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

        for sample in data_generator:
            self.a2c_update(sample, update_actor)

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()