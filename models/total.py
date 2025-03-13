from torch import nn

from models.fusionbranch import FusionBranch
from models.networks.decisionfuse import DecisionFuse
from models.networks.output_heads import OutputHeads
from models.networks.temporal_fusion import TemporalFusionModule
from models.rgbbranch import RgbBranch
from models.thermalbranch import ThermalBranch


class Total(nn.Module):
    """
    这是整体网络架构.
    """

    def __init__(self, opt):
        super().__init__()
        self.embedding_dim = 16
        # 预处理部分
        self.cnn_rgb = nn.Conv2d(in_channels=3, out_channels=self.embedding_dim, kernel_size=1, stride=1)
        self.cnn_t = nn.Conv2d(in_channels=3, out_channels=self.embedding_dim, kernel_size=1, stride=1)
        self.cnn_hm = nn.Conv2d(in_channels=1, out_channels=self.embedding_dim, kernel_size=1, stride=1)
        # 时序融合模块
        self.temporal_fusion = TemporalFusionModule()
        # RGB分支
        self.rgb_branch = RgbBranch(opt=opt)
        # 热成像分支
        self.thermal_branch = ThermalBranch(opt=opt)
        # 模态融合分支
        self.fusion_branch = FusionBranch(opt=opt)
        # 决策融合模块
        self.decision_fuse = DecisionFuse()
        # 输出头
        self.output_heads = OutputHeads(heads={'hm': 2, 'ltrb_amodal': 4, 'reg': 2, 'tracking': 2, 'wh': 2},
                                        head_convs={'hm': [256], 'ltrb_amodal': [256], 'reg': [256], 'tracking': [256],
                                                    'wh': [256]},
                                        num_stacks=1, last_channel=64, opt=opt)

    def forward(self, rgb, thermal, rgb_pre=None, thermal_pre=None, hm_pre=None):
        # 预处理
        rgb = self.cnn_rgb(rgb)
        thermal = self.cnn_t(thermal)

        # 时序融合
        if rgb_pre is not None and thermal_pre is not None:
            rgb_pre = self.cnn_rgb(rgb_pre)
            thermal_pre = self.cnn_t(thermal_pre)
            temporal_fusion_rgb = self.temporal_fusion(rgb, rgb_pre)
            temporal_fusion_thermal = self.temporal_fusion(thermal, thermal_pre)
        else:
            temporal_fusion_rgb = self.temporal_fusion(rgb, rgb)
            temporal_fusion_thermal = self.temporal_fusion(thermal, thermal)

        # 融合热力图
        if hm_pre is not None:
            hm_pre = self.cnn_hm(hm_pre)
            temporal_fusion_rgb += hm_pre
            temporal_fusion_thermal += hm_pre

        # 过各自分支
        rgb_branch = self.rgb_branch(temporal_fusion_rgb)[-1]
        thermal_branch = self.thermal_branch(temporal_fusion_thermal)[-1]
        modality_fusion = self.fusion_branch(temporal_fusion_rgb, temporal_fusion_thermal)[-1]

        # 决策融合
        decision_fuse = self.decision_fuse(rgb=rgb_branch, thermal=thermal_branch, fusion=modality_fusion)

        # 输出头
        output = self.output_heads(feats=decision_fuse)

        return output


if __name__ == '__main__':
    from tools.opts import opts
    import torch

    opt = opts().init()
    rgb = torch.randn(1, 3, 544, 960)
    thermal = torch.randn(1, 3, 544, 960)
    hm = torch.randn(1, 1, 544, 960)
    model = Total(opt)
    output = model(rgb, thermal, rgb, thermal, hm)[-1]
    for each in output:
        print(each, output[each].shape)
