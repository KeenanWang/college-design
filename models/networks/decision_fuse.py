import torch
import torch.nn.functional as F
from torch import nn

from models.networks.transformer import NonLocalBlock


class DecisionFuse(nn.Module):
    """
    将三个分支的决策融合起来。同样采用squeeze and excitation策略。
    """

    def __init__(self):
        super().__init__()
        # 降采样
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=7)
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
        assert rgb.size() == thermal.size() == fusion.size()
        _, _, h, w = rgb.shape
        # 降采样
        rgb = self.conv(rgb)
        thermal = self.conv(thermal)
        fusion = self.conv(fusion)

        # squeeze
        _, v = self.nl_v(rgb, rgb, rgb)
        _, t = self.nl_fused(thermal, thermal, thermal)
        _, f = self.nl_t(fusion, fusion, fusion)
        v = v.sum(dim=1, keepdim=True)
        t = t.sum(dim=1, keepdim=True)
        f = f.sum(dim=1, keepdim=True)

        # softmax
        v_s = torch.softmax(v, dim=1)
        t_s = torch.softmax(t, dim=1)
        f_s = torch.softmax(f, dim=1)

        # 中期fuse
        rgb_m = v_s * rgb
        thermal_m = t_s * thermal
        fusion_m = f_s * fusion

        # excitation
        squeeze = torch.cat((v, t, f), dim=1)
        excitation = self.excitation(squeeze)
        w_v, w_t, w_f = torch.split(excitation, 1, dim=1)

        rgb_fuse = rgb_m + rgb * w_v
        thermal_fuse = thermal_m + thermal * w_t
        fusion_fuse = fusion_m + fusion * w_f

        # 最后融合
        final = rgb_fuse + thermal_fuse + fusion_fuse
        final = F.interpolate(final, size=(h, w), mode='bilinear', align_corners=True)
        return final


if __name__ == "__main__":
    rgb = torch.randn(1, 64, 135, 240).to(torch.device('cuda:0'))
    thermal = torch.randn(1, 64, 135, 240).to(torch.device('cuda:0'))
    fusion = torch.randn(1, 64, 135, 240).to(torch.device('cuda:0'))
    model = DecisionFuse().to(torch.device('cuda:0'))
    print(model(rgb, thermal, fusion).shape)
    print(model(rgb, thermal, fusion).shape == fusion.shape)
