import torch
import torch.nn.functional as F
from torch import nn, sqrt, tensor
from torch.nn.functional import softmax


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V):
        """
        实现的是注意力机制，没有mask
        :param Q: tensor:b,n,d
        :param K:
        :param V:
        :return: 注意力分数，输出
        """
        d_k = Q.size(-1)
        scale = 1.0 / sqrt(tensor(d_k))
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        A = self.softmax(scores)
        return A, torch.matmul(A, V)


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
        A = softmax(torch.matmul(Q, K.transpose(-1, -2)) / self.tau,dim=-1)

        # 输出注意力结果
        M = torch.matmul(A, V).permute(0, 2, 1).contiguous()
        # 恢复形状
        M = M.view(B, C // self.down_ratio, H, W)
        M = self.conv_o(M)
        return A, M


class MultiHeadAttention(nn.Module):
    """
    自主实现的多头注意力机制。输入为已经完成embedding的QKV。
    """

    def __init__(self, n_head, d_model):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_q_new = d_model // n_head
        self.d_k_new = d_model // n_head
        self.d_v_new = d_model // n_head
        self.fc_q = nn.Linear(d_model, d_model)  # b,n,d -> b,n,h*d/h
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.attention = Attention()

    def forward(self, Q, K, V):
        batch, n_q, d_q = Q.shape
        batch, n_k, d_k = K.shape
        batch, n_v, d_v = V.shape

        # 过线性层
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        # 塑形
        Q = Q.view(batch, n_q, self.n_head, self.d_q_new).permute(0, 2, 1, 3).contiguous().view(-1, n_q, self.d_q_new)
        K = K.view(batch, n_k, self.n_head, self.d_k_new).permute(0, 2, 1, 3).contiguous().view(-1, n_k, self.d_k_new)
        V = V.view(batch, n_v, self.n_head, self.d_v_new).permute(0, 2, 1, 3).contiguous().view(-1, n_v, self.d_v_new)

        # 多头注意力
        a, output = self.attention(Q, K, V)

        output = output.view(batch, self.n_head, n_q, self.d_q_new).permute(0, 2, 1, 3).contiguous().view(batch, n_q,
                                                                                                          self.d_model)
        # 过线性层
        output = self.fc_o(output)
        return a, output


if __name__ == '__main__':
    # 初始化模型
    d_model = 512
    n_head = 8
    your_mha = MultiHeadAttention(n_head, d_model)
    official_mha = nn.MultiheadAttention(d_model, n_head, batch_first=True)

    # 同步参数
    with torch.no_grad():
        # 官方参数分解
        q_weight = official_mha.in_proj_weight[:d_model]
        k_weight = official_mha.in_proj_weight[d_model:2 * d_model]
        v_weight = official_mha.in_proj_weight[2 * d_model:]

        # 若启用偏置，需添加：
        your_mha.fc_q.bias.copy_(official_mha.in_proj_bias[:d_model])
        your_mha.fc_k.bias.copy_(official_mha.in_proj_bias[d_model:2 * d_model])
        your_mha.fc_v.bias.copy_(official_mha.in_proj_bias[2 * d_model:])
        your_mha.fc_o.bias.copy_(official_mha.out_proj.bias)

        # 复制到您的模型
        your_mha.fc_q.weight.copy_(q_weight)
        your_mha.fc_k.weight.copy_(k_weight)
        your_mha.fc_v.weight.copy_(v_weight)
        your_mha.fc_o.weight.copy_(official_mha.out_proj.weight)

    # 前向传播测试
    Q = torch.randn(2, 10, 512)
    your_attn, your_output = your_mha(Q, Q, Q)
    official_output, _ = official_mha(Q, Q, Q)
    print(your_output, official_output)
    print(torch.allclose(your_output, official_output, atol=1e-6))  # 应输出True
