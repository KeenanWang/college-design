import torch
from torch import nn

from models.networks.attention import NonLocalBlock


class DecisionFuse(nn.Module):
    """
    将三个分支的决策融合起来。同样采用squeeze and excitation策略。
    """

    def __init__(self):
        super().__init__()
        # squeeze部分
        self.nl_v = NonLocalBlock(in_channels=64)
        self.nl_fused = NonLocalBlock(in_channels=64)
        self.nl_t = NonLocalBlock(in_channels=64)

        # excitation部分
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, rgb, thermal, fusion):
        # squeeze
        _, v = self.nl_v(rgb, rgb, rgb)
        _, t = self.nl_fused(thermal, thermal, thermal)
        _, f = self.nl_t(fusion, fusion, fusion)
        v = v.sum(dim=1, keepdim=True)
        t = t.sum(dim=1, keepdim=True)
        f = f.sum(dim=1, keepdim=True)

        # 中期fuse
        rgb_m = v * rgb
        thermal_m = t * thermal
        fusion_m = f * fusion

        # excitation
        squeeze = torch.cat((v, t, f), dim=1)
        excitation = self.excitation(squeeze)
        w_v, w_t, w_f = torch.split(excitation, 1, dim=1)
        final = rgb_m * w_v + thermal_m * w_t + fusion_m * w_f
        return final


if __name__ == "__main__":
    rgb = torch.randn(1, 64, 135, 240).to(torch.device('cuda:0'))
    thermal = torch.randn(1, 64, 135, 240).to(torch.device('cuda:0'))
    fusion = torch.randn(1, 64, 135, 240).to(torch.device('cuda:0'))
    model = DecisionFuse().to(torch.device('cuda:0'))
    print(model(rgb, thermal, fusion).shape)
