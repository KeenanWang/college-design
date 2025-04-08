import torch
import torch.nn as nn


class ModalityMix(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # rgb提取
        self.fc_v = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.LayerNorm(in_dims),
            nn.ReLU(),
        )

        # 融合
        self.fc_mix = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.LayerNorm(in_dims),
            nn.ReLU(),
        )
        self.fc_mix_v = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.LayerNorm(in_dims),
            nn.ReLU(),
        )
        self.fc_mix_t = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.LayerNorm(in_dims),
            nn.ReLU(),
        )

        # thermal提取
        self.fc_t = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.LayerNorm(in_dims),
            nn.ReLU(),
        )

    def forward(self, rgb, thermal):
        # rgb分支
        rgb_v = self.avg_pool(rgb).squeeze(-1).squeeze(-1)
        rgb_v = self.fc_v(rgb_v)
        rgb_v = torch.softmax(rgb_v, dim=1).unsqueeze(-1).unsqueeze(-1) * rgb

        # thermal分支
        thermal_t = self.avg_pool(thermal).squeeze(-1).squeeze(-1)
        thermal_t = self.fc_t(thermal_t)
        thermal_t = torch.softmax(thermal_t, dim=1).unsqueeze(-1).unsqueeze(-1) * thermal

        # fuse分支
        # Squeeze: 全局平均池化
        squeeze = self.avg_pool(rgb + thermal)  # [1,3,1,1]
        squeeze = squeeze.squeeze(-1).squeeze(-1)  # [1,3]

        # Excitation: 全连接层
        fc = self.fc_mix(squeeze)  # [1,3]
        v = self.fc_mix_v(fc)  # [1,3]
        t = self.fc_mix_t(fc)  # [1,3]

        # 拼接并计算权重
        combined = torch.cat([v.unsqueeze(1), t.unsqueeze(1)], dim=1)  # [1,2,3]
        weights = torch.softmax(combined, dim=1)  # 对通道维度做softmax
        v_weight, t_weight = weights[:, 0, :], weights[:, 1, :]  # [1,3]

        # 调整权重形状以匹配特征图
        v_weight = v_weight.unsqueeze(-1).unsqueeze(-1)  # [1,3,1,1]
        t_weight = t_weight.unsqueeze(-1).unsqueeze(-1)  # [1,3,1,1]

        # 加权融合
        rgb_fuse = rgb * v_weight + rgb_v  # [1,3,544,960]
        thermal_fuse = thermal * t_weight + thermal_t

        # 最后融合
        final = rgb_fuse + thermal_fuse
        return final


if __name__ == '__main__':
    rgb = torch.randn(1, 3, 544, 960)
    thermal = torch.randn(1, 3, 544, 960)
    model = ModalityMix(16)
    print(model(rgb, thermal).shape)  # 输出: torch.Size([1, 3, 544, 960])
