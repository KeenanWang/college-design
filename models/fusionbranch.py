from torch import nn

from models.networks.backbones.dla import DLASeg
from models.networks.modality_mix import ModalityMix


class FusionBranch(nn.Module):
    """
    这是融合分支。
    """

    def __init__(self, opt):
        super().__init__()
        # 降维
        self.cnn_down = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        # 模态融合器
        self.modality_mix = ModalityMix()
        # DLA网络
        self.dcn = DLASeg(num_layers=34, heads={'hm': 2, 'ltrb_amodal': 4, 'reg': 2, 'tracking': 2, 'wh': 2},
                          head_convs={'hm': [256], 'ltrb_amodal': [256], 'reg': [256], 'tracking': [256], 'wh': [256]},
                          opt=opt)

    def forward(self, rgb, thermal):
        rgb_down = self.cnn_down(rgb)
        thermal_down = self.cnn_down(thermal)
        fuse = self.modality_mix(rgb_down, thermal_down)
        final = self.dcn(fuse)
        return final


if __name__ == '__main__':
    import torch
    from tools.opts import opts

    opt = opts().init()
    rgb = torch.randn(1, 16, 544, 960)
    thermal = torch.randn(1, 16, 544, 960)
    model = FusionBranch(opt=opt)
    print(model(rgb, thermal))
