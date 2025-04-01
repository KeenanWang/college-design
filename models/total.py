from torch import nn

from models.fusionbranch import FusionBranch
from models.networks.decision_fuse import DecisionFuse
from models.networks.temporal_fusion import TemporalFusionModule
from models.rgbbranch import RgbBranch
from models.thermalbranch import ThermalBranch


class Total(nn.Module):
    """
    这是整体网络架构.
    """

    def __init__(self, opt):
        super().__init__()
        self.embedding_dim = opt.embedding_dim
        self.kernel_stride = opt.kernel_stride
        # 预处理部分
        self.cnn_rgb = nn.Conv2d(in_channels=3, out_channels=self.embedding_dim, kernel_size=1, stride=1)
        self.cnn_t = nn.Conv2d(in_channels=3, out_channels=self.embedding_dim, kernel_size=1, stride=1)
        self.cnn_hm = nn.Conv2d(in_channels=1, out_channels=self.embedding_dim, kernel_size=1, stride=1)
        # 时序融合模块
        self.temporal_fusion_rgb = TemporalFusionModule(h=opt.input_h, w=opt.input_w, embedding_dim=self.embedding_dim,
                                                        kernel_stride=self.kernel_stride)
        self.temporal_fusion_thermal = TemporalFusionModule(h=opt.input_h, w=opt.input_w,
                                                            embedding_dim=self.embedding_dim,
                                                            kernel_stride=self.kernel_stride)
        # RGB分支
        self.rgb_branch = RgbBranch(opt=opt)
        # 热成像分支
        self.thermal_branch = ThermalBranch(opt=opt)
        # 模态融合分支
        self.fusion_branch = FusionBranch(opt=opt)
        # 决策融合模块
        self.decision_fuse = DecisionFuse(c=2, h=opt.input_h // opt.down_ratio, w=opt.input_w // opt.down_ratio)

    def forward(self, rgb, thermal, rgb_pre=None, thermal_pre=None, hm_pre=None):
        # 预处理
        rgb_processed = self.cnn_rgb(rgb)
        thermal_processed = self.cnn_t(thermal)

        # 融合热力图
        if hm_pre is not None:
            hm_pre = self.cnn_hm(hm_pre)

        # 时序融合
        if rgb_pre is not None and thermal_pre is not None:
            rgb_pre = self.cnn_rgb(rgb_pre)
            thermal_pre = self.cnn_t(thermal_pre)
            temporal_fusion_rgb = self.temporal_fusion_rgb(rgb_processed, rgb_pre, hm_pre)
            temporal_fusion_thermal = self.temporal_fusion_thermal(thermal_processed, thermal_pre, hm_pre)
        else:
            temporal_fusion_rgb = self.temporal_fusion_rgb(rgb_processed, rgb_processed, hm_pre)
            temporal_fusion_thermal = self.temporal_fusion_thermal(thermal_processed, thermal_processed, hm_pre)

        # 过各自分支
        rgb_branch = self.rgb_branch(temporal_fusion_rgb)[-1]
        thermal_branch = self.thermal_branch(temporal_fusion_thermal)[-1]
        modality_fusion = self.fusion_branch(temporal_fusion_rgb, temporal_fusion_thermal)[-1]

        # 决策融合
        final = self.decision_fuse(rgb_branch, thermal_branch, modality_fusion)
        return [rgb_branch], [thermal_branch], [final]


if __name__ == '__main__':
    from utils.opts import opts
    import torch

    opt = opts().parse()
    rgb = torch.randn(1, 3, 544, 960).cuda()
    thermal = torch.randn(1, 3, 544, 960).cuda()
    hm = torch.randn(1, 1, 544, 960).cuda()
    opt.input_h = 544
    opt.input_w = 960
    model = Total(opt).to('cuda')
    output = model(rgb, thermal, rgb, thermal, hm)[-1][-1]
    for each in output:
        print(each, output[each].shape)
