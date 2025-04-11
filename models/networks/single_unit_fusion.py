import torch
from torch import nn


class SingleUnitFusion(nn.Module):
    def __init__(self, num_branch, c, h, w):
        super().__init__()
        self.num_branch = num_branch
        self.fc = nn.Sequential(
            nn.Linear(num_branch * c * h * w, 16),
            nn.ReLU(),
            nn.Linear(16, num_branch),
            nn.Softmax(dim=1),
        )

    def forward(self, rgb, thermal, fusion, type_choose):
        b_x, c_x, h_x, w_x = rgb[type_choose].shape
        x_rgb, x_thermal, x_fusion = rgb[type_choose], thermal[type_choose], fusion[type_choose]

        x = torch.cat([x_rgb, x_thermal, x_fusion], dim=1)
        x_flattened = torch.flatten(x, 1)

        x_rgb_weight, x_thermal_weight, x_fusion_weight = torch.split(
            self.fc(x_flattened).view(b_x, self.num_branch, 1, 1), 1, 1)

        x_output = x_rgb_weight * x_rgb + x_thermal_weight * x_thermal + x_fusion_weight * x_fusion
        return x_output
