from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.

    本版本兼容四类动作空间：
    1. Discrete
    2. Box
    3. MultiBinary
    4. MultiDiscrete
    5. Tuple(MultiDiscrete + Box) 混合动作空间

    kill_chain 连续速度版本使用：
        Tuple((MultiDiscrete([3, max_candidates]), Box(low=-1, high=1, shape=(3,))))

    Actor 输出的扁平动作格式固定为：
        [strategy, candidate_idx, vx, vy, vz]

    其中 strategy/candidate_idx 为离散动作，vx/vy/vz 为连续高斯动作输出。
    """

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.discrete_action = False
        self.box_action = False
        self.multi_binary = False

        if action_space.__class__.__name__ == "Discrete":
            self.discrete_action = True
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)

        elif action_space.__class__.__name__ == "Box":
            self.box_action = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)

        elif action_space.__class__.__name__ == "MultiBinary":
            self.multi_binary = True
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)

        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True

            if hasattr(action_space, 'nvec'):
                action_dims = action_space.nvec
            else:
                action_dims = action_space.high - action_space.low + 1

            self.action_dims = list(action_dims)
            self.action_outs = nn.ModuleList([
                Categorical(inputs_dim, action_dim, use_orthogonal, gain)
                for action_dim in self.action_dims
            ])

        elif action_space.__class__.__name__ == "Tuple":
            self.mixed_action = True

            self.mixed_discrete_dims = []
            self.continuous_dim = 0

            # 为了环境解析简单，本层统一输出：离散动作在前，连续动作在后。
            for subspace in action_space.spaces:
                name = subspace.__class__.__name__
                if name == "Discrete":
                    self.mixed_discrete_dims.append(subspace.n)
                elif name == "MultiDiscrete":
                    self.mixed_discrete_dims.extend(list(subspace.nvec))
                elif name == "Box":
                    self.continuous_dim += int(subspace.shape[0])
                else:
                    raise NotImplementedError(f"Unsupported Tuple subspace: {subspace}")

            if len(self.mixed_discrete_dims) == 0 or self.continuous_dim <= 0:
                raise ValueError("Tuple mixed action space must contain at least one discrete head and one Box head.")

            self.discrete_outs = nn.ModuleList([
                Categorical(inputs_dim, action_dim, use_orthogonal, gain)
                for action_dim in self.mixed_discrete_dims
            ])
            self.continuous_out = DiagGaussian(inputs_dim, self.continuous_dim, use_orthogonal, gain)

        else:
            raise NotImplementedError(f"Unsupported action space: {action_space}")

    def forward(self, x, available_actions=None, deterministic=False):
        """Compute actions and action logprobs from given input."""
        if self.mixed_action:
            discrete_actions = []
            discrete_log_probs = []
            split_masks = self._split_mixed_available_actions(available_actions)

            for action_out, sub_mask in zip(self.discrete_outs, split_masks):
                action_dist = action_out(x, sub_mask)
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                discrete_actions.append(action.float())
                discrete_log_probs.append(action_log_prob)

            continuous_dist = self.continuous_out(x)
            continuous_action = continuous_dist.mode() if deterministic else continuous_dist.sample()

            # 对连续动作保留每个维度的 log_prob，使 action_log_probs 与扁平 action 同维度。
            # PPO 更新时会在最后一维求和。
            continuous_log_probs = continuous_dist.log_prob(continuous_action)

            actions = torch.cat(discrete_actions + [continuous_action], dim=-1)
            action_log_probs = torch.cat(discrete_log_probs + [continuous_log_probs], dim=-1)

        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            split_masks = self._split_available_actions(available_actions)

            for action_out, sub_mask in zip(self.action_outs, split_masks):
                action_dist = action_out(x, sub_mask)
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)

        elif self.box_action:
            action_dist = self.action_out(x)
            actions = action_dist.mode() if deterministic else action_dist.sample()
            # Box 动作也保留逐维 log_prob，避免连续多维动作被错误压成 1 维。
            action_log_probs = action_dist.log_prob(actions)

        elif self.discrete_action:
            action_dist = self.action_out(x, available_actions)
            actions = action_dist.mode() if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_probs(actions)

        elif self.multi_binary:
            action_dist = self.action_out(x)
            actions = action_dist.mode() if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_probs(actions)

        else:
            raise RuntimeError("Invalid ACTLayer action mode.")

        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        """Compute action probabilities from inputs."""
        if self.mixed_action:
            probs = []
            split_masks = self._split_mixed_available_actions(available_actions)
            for action_out, sub_mask in zip(self.discrete_outs, split_masks):
                action_dist = action_out(x, sub_mask)
                probs.append(action_dist.probs)
            # 连续动作没有离散概率表，这里返回高斯均值，主要用于调试/可视化。
            continuous_dist = self.continuous_out(x)
            probs.append(continuous_dist.mean)
            action_probs = torch.cat(probs, -1)

        elif self.multi_discrete:
            action_probs = []
            split_masks = self._split_available_actions(available_actions)
            for action_out, sub_mask in zip(self.action_outs, split_masks):
                action_dist = action_out(x, sub_mask)
                action_probs.append(action_dist.probs)
            action_probs = torch.cat(action_probs, -1)

        elif self.box_action:
            action_dist = self.action_out(x)
            action_probs = action_dist.mean

        elif self.discrete_action:
            action_dist = self.action_out(x, available_actions)
            action_probs = action_dist.probs

        elif self.multi_binary:
            action_dist = self.action_out(x)
            action_probs = action_dist.probs

        else:
            raise RuntimeError("Invalid ACTLayer action mode.")

        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """Compute log probability and entropy of given actions."""
        if self.mixed_action:
            num_discrete = len(self.mixed_discrete_dims)
            discrete_action = action[:, :num_discrete].long()
            continuous_action = action[:, num_discrete:num_discrete + self.continuous_dim]

            action_log_probs = []
            dist_entropy_terms = []
            split_masks = self._split_mixed_available_actions(available_actions)

            for k, (action_out, sub_mask) in enumerate(zip(self.discrete_outs, split_masks)):
                action_dist = action_out(x, sub_mask)
                act_k = discrete_action[:, k:k + 1]
                action_log_probs.append(action_dist.log_probs(act_k))
                ent = action_dist.entropy()
                if active_masks is not None:
                    ent = (ent * active_masks.squeeze(-1)).sum() / active_masks.sum()
                else:
                    ent = ent.mean()
                dist_entropy_terms.append(ent)

            continuous_dist = self.continuous_out(x)
            continuous_log_probs = continuous_dist.log_prob(continuous_action)
            action_log_probs.append(continuous_log_probs)

            cont_ent = continuous_dist.entropy()
            if active_masks is not None:
                cont_ent = (cont_ent * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                cont_ent = cont_ent.mean()
            dist_entropy_terms.append(cont_ent)

            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = sum(dist_entropy_terms) / len(dist_entropy_terms)

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            split_masks = self._split_available_actions(available_actions)

            for action_out, act, sub_mask in zip(self.action_outs, action, split_masks):
                action_dist = action_out(x, sub_mask)
                action_log_probs.append(action_dist.log_probs(act.long()))
                if active_masks is not None:
                    dist_entropy.append((action_dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_dist.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = sum(dist_entropy) / len(dist_entropy)

        elif self.box_action:
            action_dist = self.action_out(x)
            action_log_probs = action_dist.log_prob(action)
            ent = action_dist.entropy()
            if active_masks is not None:
                dist_entropy = (ent * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = ent.mean()

        elif self.discrete_action:
            action_dist = self.action_out(x, available_actions)
            action_log_probs = action_dist.log_probs(action.long())
            if active_masks is not None:
                dist_entropy = (action_dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_dist.entropy().mean()

        elif self.multi_binary:
            action_dist = self.action_out(x)
            action_log_probs = action_dist.log_probs(action)
            ent = action_dist.entropy()
            if active_masks is not None:
                dist_entropy = (ent * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = ent.mean()

        else:
            raise RuntimeError("Invalid ACTLayer action mode.")

        return action_log_probs, dist_entropy

    def _split_available_actions(self, available_actions):
        if available_actions is None:
            return [None] * len(self.action_dims)

        masks = []
        start = 0
        for dim in self.action_dims:
            masks.append(available_actions[:, start:start + dim])
            start += dim
        return masks

    def _split_mixed_available_actions(self, available_actions):
        """
        混合动作空间中，available_actions 只对应离散动作头：
            [strategy_mask(3), candidate_mask(max_candidates)]
        连续速度 vx/vy/vz 不做 mask。
        """
        if available_actions is None:
            return [None] * len(self.mixed_discrete_dims)

        masks = []
        start = 0
        for dim in self.mixed_discrete_dims:
            masks.append(available_actions[:, start:start + dim])
            start += dim
        return masks
