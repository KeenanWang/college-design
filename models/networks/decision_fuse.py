import torch
from torch import nn

from models.networks.embedding_layer import EmbeddingLayer
from models.networks.single_unit_fusion import SingleUnitFusion


class DecisionFuse(nn.Module):
    """
    将三个分支的决策融合起来。
    """

    def __init__(self, ):
        super().__init__()
        # 全连接层，融合器
        self.cnn = nn.Conv2d(in_channels=192, out_channels=16, kernel_size=8, stride=8)
        self.fc = nn.Sequential(
            nn.Linear(in_features=8160, out_features=3),
            nn.LayerNorm(3),
            nn.ReLU(),
            nn.Softmax(dim=1),
        )

    def forward(self, rgb, thermal, fusion, rgb_dla, thermal_dla, fusion_dla):
        cat = torch.cat((rgb_dla, thermal_dla, fusion_dla), dim=1)
        down = self.cnn(cat)
        down = down.view(down.size(0), -1)
        rgb_weight, thermal_weight, fusion_weight = torch.split(self.fc(down), 1, dim=1)

        rgb_weight = rgb_weight.unsqueeze(-1).unsqueeze(-1)
        thermal_weight = thermal_weight.unsqueeze(-1).unsqueeze(-1)
        fusion_weight = fusion_weight.unsqueeze(-1).unsqueeze(-1)
        final = {
            'hm': rgb_weight * rgb['hm'] + thermal_weight * thermal['hm'] + fusion_weight * fusion['hm'],
            'ltrb_amodal': rgb_weight * rgb['ltrb_amodal'] + thermal_weight * thermal['ltrb_amodal'] + fusion_weight *
                           fusion['ltrb_amodal'],
            'reg': rgb_weight * rgb['reg'] + thermal_weight * thermal['reg'] + fusion_weight * fusion['reg'],
            'tracking': rgb_weight * rgb['tracking'] + thermal_weight * thermal['tracking'] + fusion_weight * fusion[
                'tracking'],
            'wh': rgb_weight * rgb['wh'] + thermal_weight * thermal['wh'] + fusion_weight * fusion['wh']
        }

        return final
