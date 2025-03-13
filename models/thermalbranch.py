import torch
import torch.nn.functional as F
from torch import nn

from models.networks.attention import MultiHeadAttention
from models.networks.backbones.dla import DLASeg
from models.networks.embedding_layer import EmbeddingLayer


class ThermalBranch(nn.Module):
    """
    这是thermal分支。
    """

    def __init__(self, opt, embedding_dim=16):
        super().__init__()
        self.num_patches = 34 * 60
        self.num_tokens = 0
        self.embedding_dim = embedding_dim

        # patch embedding
        self.embedding_layer = EmbeddingLayer(embed_dim=self.embedding_dim)
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.embedding_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        # 时序特征融合
        self.temporal_mix = MultiHeadAttention(n_head=4, d_model=self.embedding_dim)
        # 降维
        self.channel_down = nn.Conv2d(in_channels=self.embedding_dim, out_channels=3, kernel_size=1, stride=1)
        # DLA34深度特征提取网络
        self.dcn = DLASeg(num_layers=34, heads={'hm': 2, 'ltrb_amodal': 4, 'reg': 2, 'tracking': 2, 'wh': 2},
                          head_convs={'hm': [256], 'ltrb_amodal': [256], 'reg': [256], 'tracking': [256], 'wh': [256]},
                          opt=opt)

    def forward(self, x, x_pre, hm_pre):
        b, c, h, w = x.shape
        # 预处理，增加通道数，其他不变
        # x = self.cnn_rgb(x)
        # x_pre = self.cnn_rgb(x_pre)
        # hm_pre = self.cnn_hm(hm_pre)
        # patch embedding
        x = self.embedding_layer(x)
        x_pre = self.embedding_layer(x_pre)
        # 位置编码添加
        x += self.pos_embedding
        x_pre += self.pos_embedding
        # 时序特征融合
        t_a, t_mix = self.temporal_mix(x, x_pre, x_pre)
        # 解码为原来的形状
        t_mix = t_mix.permute(0, 2, 1).view(b, self.embedding_dim, h // self.embedding_dim,
                                            w // self.embedding_dim).contiguous()
        t_mix = F.interpolate(t_mix, size=(h, w), mode='bilinear', align_corners=True) + hm_pre  # hm图融合
        # 恢复到3通道
        t_mix = self.channel_down(t_mix)
        # 过DLA
        final = self.dcn(t_mix)
        return final


if __name__ == '__main__':
    from tools.opts import opts

    opt = opts().init()
    images = torch.randn(4, 16, 544, 960)
    hm = torch.randn(4, 16, 544, 960)
    net = ThermalBranch(opt=opt)
    net.train()
    print(net(images, images, hm))
