# 文件路径: onpolicy/algorithms/utils/attention_base.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import init


class EntityAttentionLayer(nn.Module):
    """
    基于 Multi-Head Attention 的实体特征提取层
    输入: [Batch, Num_Entities, Feat_Dim]
    输出: [Batch, Num_Entities, Hidden_Dim]
    """

    def __init__(self, input_dim, hidden_dim, num_heads=1, use_orthogonal=True):
        super(EntityAttentionLayer, self).__init__()

        # 1. 特征映射 (Embedding)
        self.feature_encoder = nn.Linear(input_dim, hidden_dim)

        # 2. Multi-Head Self-Attention
        # batch_first=True 使得输入格式为 (Batch, Seq, Feature)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # 3. LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)

        # 初始化
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.feature_encoder = init_(self.feature_encoder)

    def forward(self, x, valid_mask=None):
        # x: [Batch, N, In_Dim]
        x_emb = self.feature_encoder(x)

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()

            # 防止某些 batch 全是 padding，MultiheadAttention 出 NaN
            if key_padding_mask.ndim == 2:
                empty_rows = key_padding_mask.all(dim=1)
                if empty_rows.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[empty_rows, 0] = False

        attn_output, _ = self.attn(
            x_emb, x_emb, x_emb,
            key_padding_mask=key_padding_mask
        )

        output = self.norm(x_emb + attn_output)
        return output


class AttentionBase(nn.Module):
    """
    针对 KillChain 环境定制的 Attention 特征提取器
    替代原本的 MLPBase
    """

    def __init__(self, args, obs_shape):
        super(AttentionBase, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal

        # Obs: [Self(10) + Buffer(7) + Compass(6) + Stats(6) = 29] + Tasks(N*7) + Neighbors(5*7)
        obs_dim = obs_shape[0]
        self.self_dim = 32
        self.task_feat_dim = 7
        self.neighbor_feat_dim = 7
        self.n_neighbors = 5

        neighbor_part_len = self.n_neighbors * self.neighbor_feat_dim
        task_part_len = obs_dim - self.self_dim - neighbor_part_len
        self.n_tasks = task_part_len // self.task_feat_dim

        if task_part_len % self.task_feat_dim != 0:
            print(
                f"[AttentionBase] Warning: Obs dim {obs_dim} parsing failed. "
                f"Self={self.self_dim}, Neigh={neighbor_part_len}, Remain={task_part_len}."
            )

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.self_encoder = nn.Sequential(
            init_(nn.Linear(self.self_dim, self.hidden_size)),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )

        self.task_attn_layer = EntityAttentionLayer(
            self.task_feat_dim, self.hidden_size,
            num_heads=1,
            use_orthogonal=self._use_orthogonal
        )

        self.neighbor_attn_layer = EntityAttentionLayer(
            self.neighbor_feat_dim, self.hidden_size,
            num_heads=1,
            use_orthogonal=self._use_orthogonal
        )

        self.fusion_layer = nn.Sequential(
            init_(nn.Linear(self.hidden_size * 3, self.hidden_size)),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]

        self_inputs = obs[:, :self.self_dim]

        task_start = self.self_dim
        task_end = task_start + (self.n_tasks * self.task_feat_dim)
        task_flat = obs[:, task_start:task_end]
        task_inputs = task_flat.reshape(batch_size, self.n_tasks, self.task_feat_dim)

        neighbor_start = task_end
        neighbor_flat = obs[:, neighbor_start:]
        neighbor_inputs = neighbor_flat.reshape(batch_size, self.n_neighbors, self.neighbor_feat_dim)

        self_feat = self.self_encoder(self_inputs)

        # task valid: 第 0 维是 valid 位
        task_valid = task_inputs[..., 0] > 0.5
        task_emb = self.task_attn_layer(task_inputs, valid_mask=task_valid)
        task_emb = task_emb.masked_fill(~task_valid.unsqueeze(-1), -1e9)
        task_pool = task_emb.max(dim=1).values
        task_any = task_valid.any(dim=1, keepdim=True)
        task_pool = torch.where(task_any, task_pool, torch.zeros_like(task_pool))

        # neighbor valid: 非全 0
        neighbor_valid = (neighbor_inputs.abs().sum(dim=-1) > 1e-6)
        neighbor_emb = self.neighbor_attn_layer(neighbor_inputs, valid_mask=neighbor_valid)
        neighbor_emb = neighbor_emb.masked_fill(~neighbor_valid.unsqueeze(-1), -1e9)
        neighbor_pool = neighbor_emb.max(dim=1).values
        neighbor_any = neighbor_valid.any(dim=1, keepdim=True)
        neighbor_pool = torch.where(neighbor_any, neighbor_pool, torch.zeros_like(neighbor_pool))

        concat_feat = torch.cat([self_feat, task_pool, neighbor_pool], dim=1)
        final_feat = self.fusion_layer(concat_feat)

        return final_feat