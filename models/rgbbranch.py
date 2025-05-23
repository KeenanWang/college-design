import torch
from torch import nn

from models.networks.backbones.dla import DLASeg
from models.networks.output_heads import OutputHeads


class RgbBranch(nn.Module):
    """
    这是RGB分支。从DLA开始
    """

    def __init__(self, opt, embedding_dim=16):
        super().__init__()
        self.embedding_dim = embedding_dim
        # 降维
        self.channel_down = nn.Conv2d(in_channels=self.embedding_dim, out_channels=3, kernel_size=1, stride=1)
        # dla
        self.dla = DLASeg(num_layers=34, heads={'hm': 2, 'ltrb_amodal': 4, 'reg': 2, 'tracking': 2, 'wh': 2},
                          head_convs={'hm': [256], 'ltrb_amodal': [256], 'reg': [256], 'tracking': [256], 'wh': [256]},
                          opt=opt)
        # 输出头
        self.output_heads = OutputHeads(heads={'hm': 2, 'ltrb_amodal': 4, 'reg': 2, 'tracking': 2, 'wh': 2},
                                        head_convs={'hm': [256], 'ltrb_amodal': [256], 'reg': [256], 'tracking': [256],
                                                    'wh': [256]},
                                        num_stacks=1, last_channel=64, opt=opt)

    def forward(self, temporal_fusion):
        # 恢复到3通道
        t_mix = self.channel_down(temporal_fusion)
        # 过DLA
        dla_output = self.dla(t_mix)
        # 输出头
        final = self.output_heads(dla_output)
        return final, dla_output


if __name__ == '__main__':
    from utils.opts import opts

    opt = opts().parse()
    images = torch.randn(4, 16, 544, 960)
    net = RgbBranch(opt=opt)
    output = net(images)
    for each in output[0]:
        print(each, output[0][each].shape)
