import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import softmax


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, down_ratio=4, tau=30):
        super().__init__()
        self.tau = tau  # 温度参数τ设为30
        self.down_ratio = down_ratio
        # 卷积层
        self.conv_q = nn.Conv2d(in_channels, in_channels // self.down_ratio, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels // self.down_ratio, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels // self.down_ratio, kernel_size=1)
        self.conv_o = nn.Conv2d(in_channels // self.down_ratio, in_channels, kernel_size=1)

    def forward(self, Q, K, V):
        B, C, H, W = Q.size()

        # 过卷积层
        Q = self.conv_q(Q)
        K = self.conv_k(K)
        V = self.conv_v(V)

        # 改变形状
        Q = Q.view(B, C // self.down_ratio, H * W).permute(0, 2, 1).contiguous()
        K = K.view(B, C // self.down_ratio, H * W).permute(0, 2, 1).contiguous()
        V = V.view(B, C // self.down_ratio, H * W).permute(0, 2, 1).contiguous()

        # L-2归一化
        Q = F.normalize(Q, p=2, dim=1)  # 特征维度归一化
        K = F.normalize(K, p=2, dim=1)  # 通道维度归一化

        # 计算注意力分数
        A = softmax(torch.matmul(Q, K.transpose(-1, -2)) / self.tau, dim=-1)

        # 输出注意力结果
        M = torch.matmul(A, V).permute(0, 2, 1).contiguous()
        # 恢复形状
        M = M.view(B, C // self.down_ratio, H, W)
        M = self.conv_o(M)
        return A, M


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, x, x_pre, hm_pre):
        for layer in self.layers:
            x = layer(x, x_pre, hm_pre)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, normalize_before=False):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = nn.Dropout(dropout)

        self.activation = activation

    def with_pos_embed(self, tensor, pos=None):
        return tensor + pos if pos is not None else tensor

    def forward(self, x, x_pre, hm_pre):
        q = x
        k = x_pre
        x_att = self.multihead_attention(query=q, key=k, value=x_pre)[0]  # 提取注意力分数，【1】是注意力权重
        x = x + self.dropout(x_att)
        x = self.norm1(x)

        x_mix_hm = x + hm_pre

        x_new = self.linear2(self.dropout1(self.activation(self.linear1(x_mix_hm))))
        x = x + self.dropout2(x_new)
        x = self.norm2(x)
        return x
