from torch import nn

from models.fusionbranch import FusionBranch
from models.networks.backbones.dla import DLASeg
from models.networks.decision_fuse import DecisionFuse
from models.networks.output_heads import OutputHeads
from models.networks.temporal_fusion import TemporalFusionModule
from models.rgbbranch import RgbBranch
from models.thermalbranch import ThermalBranch


class Total_MDP(nn.Module):
    """
    这是整体网络架构，模型并行版。
    """

    def __init__(self, opt):
        super().__init__()
        self.embedding_dim = 16
        # 预处理部分，放到0号GPU
        self.cnn_rgb = nn.Conv2d(in_channels=3, out_channels=self.embedding_dim, kernel_size=1, stride=1).to('cuda:0')
        self.cnn_t = nn.Conv2d(in_channels=3, out_channels=self.embedding_dim, kernel_size=1, stride=1).to('cuda:0')
        self.cnn_hm = nn.Conv2d(in_channels=1, out_channels=self.embedding_dim, kernel_size=1, stride=1).to('cuda:0')
        # 时序融合模块，放到1号GPU
        self.temporal_fusion = TemporalFusionModule().to('cuda:1')
        # RGB分支，2号
        self.rgb_branch = RgbBranch(opt=opt).to('cuda:2')
        # 热成像分支，2号
        self.thermal_branch = ThermalBranch(opt=opt).to('cuda:2')
        # 模态融合分支，2号
        self.fusion_branch = FusionBranch(opt=opt).to('cuda:2')
        # DLA34深度特征提取网络,3
        self.dla = DLASeg(num_layers=34, heads={'hm': 2, 'ltrb_amodal': 4, 'reg': 2, 'tracking': 2, 'wh': 2},
                          head_convs={'hm': [256], 'ltrb_amodal': [256], 'reg': [256], 'tracking': [256], 'wh': [256]},
                          opt=opt)
        # 遍历所有参数，冻结dla网络
        for name, param in self.dla.named_parameters():
            param.requires_grad = False
        self.dla = self.dla.to('cuda:3')
        # 决策融合模块，4号
        self.decision_fuse = DecisionFuse().to('cuda:4')
        # 输出头，5号
        self.output_heads = OutputHeads(heads={'hm': 2, 'ltrb_amodal': 4, 'reg': 2, 'tracking': 2, 'wh': 2},
                                        head_convs={'hm': [256], 'ltrb_amodal': [256], 'reg': [256], 'tracking': [256],
                                                    'wh': [256]},
                                        num_stacks=1, last_channel=64, opt=opt).to('cuda:5')

    def forward(self, rgb, thermal, rgb_pre=None, thermal_pre=None, hm_pre=None):
        # 预处理
        rgb_processed = self.cnn_rgb(rgb)
        thermal_processed = self.cnn_t(thermal)
        rgb_processed = rgb_processed.to('cuda:1')
        thermal_processed = thermal_processed.to('cuda:1')

        # 时序融合
        if rgb_pre is not None and thermal_pre is not None:
            rgb_pre = self.cnn_rgb(rgb_pre).to('cuda:1')
            thermal_pre = self.cnn_t(thermal_pre).to('cuda:1')
            temporal_fusion_rgb = self.temporal_fusion(rgb_processed, rgb_pre)
            temporal_fusion_thermal = self.temporal_fusion(thermal_processed, thermal_pre)
        else:
            temporal_fusion_rgb = self.temporal_fusion(rgb_processed, rgb_processed)
            temporal_fusion_thermal = self.temporal_fusion(thermal_processed, thermal_processed)

        # 融合热力图
        if hm_pre is not None:
            hm_pre = self.cnn_hm(hm_pre).to('cuda:1')
            temporal_fusion_rgb = temporal_fusion_rgb + hm_pre
            temporal_fusion_thermal = temporal_fusion_thermal + hm_pre

        temporal_fusion_rgb = temporal_fusion_rgb.to('cuda:2')
        temporal_fusion_thermal = temporal_fusion_thermal.to('cuda:2')

        # 过各自分支
        rgb_branch = self.rgb_branch(temporal_fusion_rgb)
        thermal_branch = self.thermal_branch(temporal_fusion_thermal)
        modality_fusion = self.fusion_branch(temporal_fusion_rgb, temporal_fusion_thermal)
        rgb_branch = rgb_branch.to('cuda:3')
        thermal_branch = thermal_branch.to('cuda:3')
        modality_fusion = modality_fusion.to('cuda:3')

        # 过DLA网络
        rgb_branch_dla = self.dla(rgb_branch)[-1]
        thermal_branch_dla = self.dla(thermal_branch)[-1]
        modality_fusion_dla = self.dla(modality_fusion)[-1]
        rgb_branch_dla = rgb_branch_dla.to('cuda:4')
        thermal_branch_dla = thermal_branch_dla.to('cuda:4')
        modality_fusion_dla = modality_fusion_dla.to('cuda:4')

        # 决策融合
        decision_fuse = self.decision_fuse(rgb_branch_dla, thermal_branch_dla, modality_fusion_dla)
        # decision_fuse = checkpoint(self.decision_fuse, rgb_branch_dla, thermal_branch_dla, modality_fusion_dla)
        decision_fuse = decision_fuse.to('cuda:5')

        # 输出头
        output = self.output_heads(feats=[decision_fuse])

        return output


if __name__ == '__main__':
    from utils.opts import opts
    import torch

    opt = opts().init()
    rgb = torch.randn(1, 3, 544, 960).to('cuda:0')
    thermal = torch.randn(1, 3, 544, 960).to('cuda:0')
    hm = torch.randn(1, 1, 544, 960).to('cuda:0')
    model = Total(opt)
    output = model(rgb, thermal, rgb, thermal, hm)[-1]
    for each in output:
        print(each, output[each].shape)
