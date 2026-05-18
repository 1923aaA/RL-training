import torch
import numpy as np
from offpolicy.algorithms.mqmix.algorithm.agent_q_function import AgentQFunction
from offpolicy.algorithms.mqmix.algorithm.mq_mixer import M_QMixer
from offpolicy.utils.util import get_dim_from_space, soft_update, hard_update, onehot_from_logits


class M_QMixPolicy:
    def __init__(self, config, policy_info):
        self.args = config["args"]
        self.policy_info = policy_info

        self.act_space = policy_info['act_space']
        self.obs_space = policy_info['obs_space']
        self.share_obs_space = policy_info['share_obs_space']

        # 1. 基础维度定义
        self.obs_dim = self.obs_space.shape[0]
        self.share_obs_dim = self.share_obs_space.shape[0]

        # [关键修复] 必须定义 central_obs_dim，Runner 初始化需要用到它
        self.central_obs_dim = self.share_obs_dim

        # 2. 解析动作空间 (兼容 Discrete 和 MultiDiscrete)
        if hasattr(self.act_space, 'nvec'):
            # MultiDiscrete: [3, 10, 7] -> act_dim = 20
            self.act_dims = self.act_space.nvec.tolist()
            self.act_dim = int(sum(self.act_dims))
        else:
            # Discrete: n -> act_dim = n
            self.act_dims = [int(get_dim_from_space(self.act_space))]
            self.act_dim = self.act_dims[0]

        self.act_num = self.act_dim

        self.hidden_size = self.args.hidden_size
        self.device = torch.device("cuda:0") if self.args.cuda else torch.device("cpu")

        self.input_dim = self.obs_dim + self.act_dim

        # 3. 初始化网络
        self.q_network = AgentQFunction(self.args, self.input_dim, self.act_dim, self.device)
        self.target_q_network = AgentQFunction(self.args, self.input_dim, self.act_dim, self.device)
        self.target_critic = self.target_q_network

        self.mixer = M_QMixer(self.args, self.num_agents, self.share_obs_dim, self.device)
        self.target_mixer = M_QMixer(self.args, self.num_agents, self.share_obs_dim, self.device)

        # 4. 同步目标网络权重
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # 5. 优化器
        self.parameters = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.lr)

    @property
    def num_agents(self):
        return self.args.num_agents

    def get_q_values(self, obs_batch, prev_action_batch, rnn_states, action_batch=None):
        # 维度对齐处理
        if obs_batch.shape[2] != prev_action_batch.shape[2]:
            if obs_batch.shape[2] == 1:
                obs_batch = obs_batch.expand(-1, -1, prev_action_batch.shape[2], -1)
            elif prev_action_batch.shape[2] == 1:
                prev_action_batch = prev_action_batch.expand(-1, -1, obs_batch.shape[2], -1)

        input_batch = torch.cat((obs_batch, prev_action_batch), dim=-1)
        q_evals, new_rnn_states = self.q_network(input_batch, rnn_states)
        return q_evals, new_rnn_states

    def get_actions(self, obs, prev_action, rnn_states, available_actions=None, t_env=None, explore=False):
        # 转为 Tensor
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        prev_action = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
        rnn_states = torch.as_tensor(rnn_states, dtype=torch.float32, device=self.device)

        # 维度扩展 (如果 obs 是 batch 形式但没有 agent 维度)
        if obs.dim() > 1 and prev_action.dim() > 1:
            if obs.shape[-2] != prev_action.shape[-2]:
                if obs.shape[-2] == 1: obs = obs.expand(*obs.shape[:-2], prev_action.shape[-2], obs.shape[-1])

        input_batch = torch.cat((obs, prev_action), dim=-1)
        q_values, new_rnn_states = self.q_network(input_batch, rnn_states)

        # 计算 Epsilon
        epsilon = self.args.epsilon_start
        if t_env is not None:
            duration = self.args.epsilon_anneal_time
            if t_env < duration:
                epsilon = self.args.epsilon_start - (self.args.epsilon_start - self.args.epsilon_finish) * (
                            t_env / duration)
            else:
                epsilon = self.args.epsilon_finish

        # --- 分段动作选择 (支持 MultiDiscrete) ---
        final_actions = []

        # 1. 根据 act_dims 切分 Q 值向量
        q_splits = torch.split(q_values, self.act_dims, dim=-1)

        # 2. 根据 act_dims 切分 Available Actions Mask
        if available_actions is not None:
            available_actions = torch.as_tensor(available_actions, dtype=torch.float32, device=self.device)
            avail_splits = torch.split(available_actions, self.act_dims, dim=-1)
        else:
            avail_splits = [None] * len(self.act_dims)

        # 3. 对每一段分别选择动作
        for q_s, avail_s in zip(q_splits, avail_splits):
            q_s_curr = q_s.clone()

            # 应用 Mask
            if avail_s is not None:
                q_s_curr[avail_s == 0] = -1e10

            # Epsilon-Greedy 采样
            if explore and np.random.rand() < epsilon:
                act_s = onehot_from_logits(q_s_curr, avail_s, eps=1.0)  # 随机
            else:
                act_s = onehot_from_logits(q_s_curr, avail_s, eps=0.0)  # 贪婪

            final_actions.append(act_s)

        # 4. 拼接回完整的 One-Hot 向量
        actions = torch.cat(final_actions, dim=-1)

        return actions, new_rnn_states

    def init_hidden(self, num_agents, batch_size):
        return self.q_network.init_hidden(num_agents, batch_size)

    def prep_training(self):
        self.q_network.train()
        self.target_q_network.train()
        self.mixer.train()
        self.target_mixer.train()

    def prep_rollout(self):
        self.q_network.eval()
        self.target_q_network.eval()
        self.mixer.eval()
        self.target_mixer.eval()

    def soft_target_updates(self):
        soft_update(self.target_q_network, self.q_network, self.args.tau)
        soft_update(self.target_mixer, self.mixer, self.args.tau)

    def hard_target_updates(self):
        hard_update(self.target_q_network, self.q_network)
        hard_update(self.target_mixer, self.mixer)