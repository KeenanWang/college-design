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
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),

        )

    def forward(self, rgb, thermal, fusion):
        # squeeze
        v = self.nl_v(rgb, rgb, rgb)
        t = self.nl_fused(thermal, thermal, thermal)
        f = self.nl_t(fusion, fusion, fusion)

        pass


if __name__ == "__main__":
    rgb = torch.randn(4, 64, 135, 240)
    thermal = torch.randn(4, 64, 135, 240)
    fusion = torch.randn(4, 64, 135, 240)
    model = DecisionFuse()
    print(model(rgb, thermal, fusion))
