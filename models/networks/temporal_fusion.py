import torch
import torch.nn.functional as F
from torch import nn

from models.networks.transformer import TransformerEncoder, TransformerEncoderLayer
from models.networks.embedding_layer import EmbeddingLayer


class TemporalFusionModule(nn.Module):
    def __init__(self, h, w, embedding_dim, kernel_stride):
        super().__init__()
        assert w % kernel_stride == 0, f'w{w} % kernel_stride{kernel_stride}结果不为0'
        self.num_patches = w // kernel_stride * h // kernel_stride
        self.num_tokens = 0
        self.embedding_dim = embedding_dim
        self.kernel_stride = kernel_stride
        # patch embedding
        self.embedding_layer = EmbeddingLayer(embed_dim=self.embedding_dim, kernel_stride=self.kernel_stride)
        # position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.embedding_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        # 时序特征融合
        self.temporal_mix = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=4,
                dim_feedforward=64,
                dropout=0.1,
            ),
            num_layers=1,
            norm=nn.LayerNorm(self.embedding_dim)
        )

    def forward(self, x, x_pre, hm_pre=None):
        b, c, h, w = x.shape
        # patch embedding
        x = self.embedding_layer(x)
        x_pre = self.embedding_layer(x_pre)
        hm_pre = self.embedding_layer(hm_pre)
        # 位置编码添加
        x = x + self.pos_embedding
        x_pre = x_pre + self.pos_embedding
        # 时序特征融合
        t_mix = self.temporal_mix(x, x_pre, hm_pre)
        # 解码为原来的形状
        t_mix = t_mix.permute(0, 2, 1).view(b, self.embedding_dim, h // self.kernel_stride,
                                            w // self.kernel_stride).contiguous()
        t_mix = F.interpolate(t_mix, size=(h, w), mode='bilinear', align_corners=True)
        return t_mix


if __name__ == '__main__':
    from utils.opts import opts

    opt = opts().parse()
    images = torch.randn(4, 16, 544, 960)
    hm = torch.randn(4, 16, 544, 960)
    b, c, h, w = images.shape
    net = TemporalFusionModule(h, w, opt.embedding_dim, opt.kernel_stride)
    output = net(images, images, hm)
    print(output.shape)
