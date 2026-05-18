import torch
import torch.nn as nn


class AgentQFunction(nn.Module):
    def __init__(self, args, input_dim, act_dim, device):
        super(AgentQFunction, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self.device = device

        # 1. 特征提取层 (MLP)
        # 将输入映射到 hidden_size
        self.fc1 = nn.Linear(input_dim, self.hidden_size)
        if self._use_orthogonal:
            nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(self.fc1.bias.data, 0)

        # 2. RNN 层 (GRU)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=False)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

        # 3. 输出层 (Q Values)
        self.q_out = nn.Linear(self.hidden_size, act_dim)
        if self._use_orthogonal:
            nn.init.orthogonal_(self.q_out.weight.data, gain=1.0)
            nn.init.constant_(self.q_out.bias.data, 0)

        # 激活函数
        self.act_func = nn.ReLU()

        self.to(device)

    def forward(self, x, rnn_states):
        x = x.to(self.device)
        rnn_states = rnn_states.to(self.device)

        # MLP 处理
        x = self.act_func(self.fc1(x))

        if self.args.use_rnn_layer:
            # === 修复开始 ===
            # 情况 A: 训练模式，输入维度为 (Time, Batch, Agents, Hidden) 或 (Time, Batch, Hidden)
            if x.dim() == 4:
                T, B, N, H = x.shape
                # 合并 Batch 和 Agents 维度供 GRU 使用: (Time, Batch*Agents, Hidden)
                x_in = x.reshape(T, B * N, H)
                # RNN States 通常是 (Batch, Agents, Hidden) -> (1, Batch*Agents, Hidden)
                h_in = rnn_states.reshape(1, B * N, H)

                x_out, h_out = self.rnn(x_in, h_in)

                # 还原维度
                x = x_out.reshape(T, B, N, -1)  # (Time, Batch, Agents, Hidden)
                new_rnn_states = h_out.reshape(B, N, -1)  # (Batch, Agents, Hidden)

            # 情况 B: Rollout 模式，输入维度为 (Batch, Agents, Hidden) 或 (Batch, Hidden)
            elif x.dim() == 2 or x.dim() == 3:
                # 保持原有逻辑，处理单步
                if x.dim() == 3:  # (Batch, Agents, Hidden)
                    B, N, H = x.shape
                    x_in = x.reshape(1, B * N, H)
                    h_in = rnn_states.reshape(1, B * N, H)
                    x_out, h_out = self.rnn(x_in, h_in)
                    x = x_out.reshape(B, N, H)
                    new_rnn_states = h_out.reshape(B, N, H)
                else:  # (Batch, Hidden)
                    x_in = x.unsqueeze(0)
                    h_in = rnn_states.unsqueeze(0)
                    x_out, h_out = self.rnn(x_in, h_in)
                    x = x_out.squeeze(0)
                    new_rnn_states = h_out.squeeze(0)
            else:
                raise ValueError(f"Unknown input shape for RNN: {x.shape}")
            # === 修复结束 ===
        else:
            new_rnn_states = rnn_states

        q_values = self.q_out(x)
        return q_values, new_rnn_states

    def init_hidden(self, num_agents, batch_size):
        """
        初始化隐藏状态
        Returns:
            Zeros tensor of shape (batch_size * num_agents, hidden_size)
        """
        return torch.zeros(batch_size * num_agents, self.hidden_size).to(self.device)