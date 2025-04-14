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
        self.cnn_down = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        # 模态融合器
        self.modality_mix = ModalityMix(in_dims=16)
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
        fuse = self.modality_mix(rgb, thermal)
        fuse = self.cnn_down(fuse)
        dla_output = self.dla(fuse)
        final = self.output_heads(dla_output)
        return final, dla_output


if __name__ == '__main__':
    from utils.opts import opts
    import torch

    opt = opts().parse()
    images = torch.randn(4, 16, 544, 960)
    net = FusionBranch(opt=opt)
    output = net(images, images)
    for each in output[0]:
        print(each, output[0][each].shape)
