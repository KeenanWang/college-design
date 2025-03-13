import torch
import torch.nn as nn
from models.networks.attention import MultiHeadAttention  # 替换为您的实际模块路径


def test_mha_correctness():
    # 固定随机种子
    torch.manual_seed(42)

    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_head = 8

    # 初始化输入
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    # 初始化模型
    official_mha = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=n_head,
        dropout=0.0,
        batch_first=True
    )

    your_mha = MultiHeadAttention(n_head=n_head, d_model=d_model)

    # 同步参数（关键步骤）
    with torch.no_grad():
        # 同步QKV投影层参数
        your_mha.fc_q.weight.copy_(official_mha.in_proj_weight[:d_model])
        your_mha.fc_k.weight.copy_(official_mha.in_proj_weight[d_model:2 * d_model])
        your_mha.fc_v.weight.copy_(official_mha.in_proj_weight[2 * d_model:])

        # 同步输出投影层参数
        your_mha.fc_o.weight.copy_(official_mha.out_proj.weight)

    # 前向传播比较
    official_output, _ = official_mha(Q, K, V)
    your_attn, your_output = your_mha(Q, K, V)

    # 计算输出差异
    mse_output = torch.mean((official_output - your_output) ** 2).item()
    print(f"Output MSE: {mse_output:.2e}")  # 应小于1e-6

    # 反向传播比较
    target = torch.randn_like(official_output)

    # 官方实现反向
    official_output.retain_grad()
    loss_official = torch.mean((official_output - target) ** 2)
    loss_official.backward()
    grad_official = official_output.grad

    # 您的实现反向
    your_output.retain_grad()
    loss_your = torch.mean((your_output - target) ** 2)
    loss_your.backward()
    grad_your = your_output.grad

    # 比较梯度
    mse_grad = torch.mean((grad_official - grad_your) ** 2).item()
    print(f"Gradient MSE: {mse_grad:.2e}")  # 应小于1e-6


if __name__ == "__main__":
    test_mha_correctness()