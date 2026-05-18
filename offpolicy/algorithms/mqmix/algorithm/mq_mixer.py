import torch
import torch.nn as nn
import torch.nn.functional as F
from offpolicy.utils.util import init, to_torch


class M_QMixer(nn.Module):
    def __init__(self, args, num_agents, cent_obs_dim, device, multidiscrete_list=None):
        super(M_QMixer, self).__init__()
        self.device, self.num_agents, self.cent_obs_dim = device, num_agents, cent_obs_dim
        self.hidden_layer_dim, self.hypernet_hidden_dim = args.mixer_hidden_dim, args.hypernet_hidden_dim
        self.num_mixer_q_inps = sum(multidiscrete_list) if multidiscrete_list else num_agents

        def init_(m):
            init_method = nn.init.orthogonal_ if args.use_orthogonal else nn.init.xavier_uniform_
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # Hypernetworks
        self.hyper_w1 = nn.Sequential(init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)), nn.ReLU(),
                                      init_(nn.Linear(self.hypernet_hidden_dim,
                                                      self.num_mixer_q_inps * self.hidden_layer_dim)))
        self.hyper_w2 = nn.Sequential(init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)), nn.ReLU(),
                                      init_(nn.Linear(self.hypernet_hidden_dim, self.hidden_layer_dim)))
        self.hyper_b1 = init_(nn.Linear(self.cent_obs_dim, self.hidden_layer_dim))
        self.hyper_b2 = nn.Sequential(init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)), nn.ReLU(),
                                      init_(nn.Linear(self.hypernet_hidden_dim, 1)))
        self.to(device)

    def forward(self, agent_q_inps, states):
        agent_q_inps = to_torch(agent_q_inps).to(self.device)
        states = to_torch(states).to(self.device)
        seq_len = agent_q_inps.size(0)

        # [修复] 使用 reshape 并合并 Batch 维度
        states = states.reshape(-1, self.cent_obs_dim).float()
        agent_q_inps = agent_q_inps.reshape(-1, 1, self.num_mixer_q_inps).float()

        w1 = torch.abs(self.hyper_w1(states)).reshape(-1, self.num_mixer_q_inps, self.hidden_layer_dim)
        b1 = self.hyper_b1(states).reshape(-1, 1, self.hidden_layer_dim)
        hidden = F.elu(torch.bmm(agent_q_inps, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states)).reshape(-1, self.hidden_layer_dim, 1)
        b2 = self.hyper_b2(states).reshape(-1, 1, 1)

        # [修复] 严禁调用 .cpu()
        out = torch.bmm(hidden, w2) + b2
        return out.reshape(seq_len, -1, 1)