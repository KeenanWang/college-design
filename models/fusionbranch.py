from torch import nn

from models.networks.backbones.dla import DLASeg
from models.networks.modality_mix import ModalityMix
from models.networks.output_heads import OutputHeads


class FusionBranch(nn.Module):
    """
    这是融合分支。
    """

    def __init__(self, opt):
        super().__init__()
        # 降维
        self.cnn_down_rgb = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        self.cnn_down_thermal = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        # 模态融合器
        self.modality_mix = ModalityMix()
        # dla
        self.dla = DLASeg(num_layers=34, heads={'hm': 2, 'ltrb_amodal': 4, 'reg': 2, 'tracking': 2, 'wh': 2},
                          head_convs={'hm': [256], 'ltrb_amodal': [256], 'reg': [256], 'tracking': [256], 'wh': [256]},
                          opt=opt)
        # 输出头
        self.output_heads = OutputHeads(heads={'hm': 2, 'ltrb_amodal': 4, 'reg': 2, 'tracking': 2, 'wh': 2},
                                        head_convs={'hm': [256], 'ltrb_amodal': [256], 'reg': [256], 'tracking': [256],
                                                    'wh': [256]},
                                        num_stacks=1, last_channel=64, opt=opt)

    def forward(self, rgb, thermal):
        rgb_down = self.cnn_down_rgb(rgb)
        thermal_down = self.cnn_down_thermal(thermal)
        fuse = self.modality_mix(rgb_down, thermal_down)
        dla_output = self.dla(fuse)
        final = self.output_heads(dla_output)
        return final
