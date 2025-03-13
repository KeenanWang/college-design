import torch
import torch.nn as nn


class ModalityMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(3, 3)
        self.fc_v = nn.Linear(3, 3)
        self.fc_t = nn.Linear(3, 3)

    def forward(self, rgb, thermal):
        # Squeeze: 全局平均池化
        squeeze = self.avg_pool(rgb + thermal)  # [1,3,1,1]
        squeeze = squeeze.squeeze(-1).squeeze(-1)  # [1,3]

        # Excitation: 全连接层
        fc = self.fc_1(squeeze)  # [1,3]
        v = self.fc_v(fc)  # [1,3]
        t = self.fc_t(fc)  # [1,3]

        # 拼接并计算权重
        combined = torch.cat([v.unsqueeze(1), t.unsqueeze(1)], dim=1)  # [1,2,3]
        weights = torch.softmax(combined, dim=1)  # 对通道维度做softmax
        v_weight, t_weight = weights[:, 0, :], weights[:, 1, :]  # [1,3]

        # 调整权重形状以匹配特征图
        v_weight = v_weight.unsqueeze(-1).unsqueeze(-1)  # [1,3,1,1]
        t_weight = t_weight.unsqueeze(-1).unsqueeze(-1)  # [1,3,1,1]

        # 加权融合
        f_fuse = rgb * v_weight + thermal * t_weight  # [1,3,544,960]
        return f_fuse


if __name__ == '__main__':
    rgb = torch.randn(1, 3, 544, 960)
    thermal = torch.randn(1, 3, 544, 960)
    model = ModalityMix()
    print(model(rgb, thermal).shape)  # 输出: torch.Size([1, 3, 544, 960])
