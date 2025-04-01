from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self, embed_dim=16, kernel_stride=32, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_stride,
                              stride=kernel_stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 预处理，降低大小
        x = self.proj(x)
        # 对图像进行split和flatten，等同于patch embedding
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1).contiguous()
        return x
