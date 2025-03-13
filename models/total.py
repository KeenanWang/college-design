from torch import nn

from models.fusionbranch import FusionBranch
from models.networks.modality_mix import ModalityMix
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
        self.cnn_hm = nn.Conv2d(in_channels=1, out_channels=self.embedding_dim, kernel_size=1, stride=1)
        # RGB分支
        self.rgb_branch = RgbBranch(opt=opt)
        # 热成像分支
        self.thermal_branch = ThermalBranch(opt=opt)
        # 模态融合分支
        self.mix_branch = FusionBranch(opt=opt)
        # 模态融合模块
        self.modality_mix = ModalityMix()
        # 决策融合模块

        # 输出头

    def forward(self, x):
        pass


if __name__ == '__main__':
    from tools.opts import opts

    opt = opts().init()
