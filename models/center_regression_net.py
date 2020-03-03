import torch
import torch.nn as nn

from models.utils import create_mlp_components

__all__ = ['CenterRegressionNet']


class CenterRegressionNet(nn.Module):
    blocks = (128, 128, 256)

    def __init__(self, num_classes=3, width_multiplier=1):
        super().__init__()
        self.in_channels = 3
        self.num_classes = num_classes

        layers, channels = create_mlp_components(in_channels=self.in_channels, out_channels=self.blocks,
                                                 classifier=False, dim=2, width_multiplier=width_multiplier)
        self.features = nn.Sequential(*layers)

        layers, _ = create_mlp_components(in_channels=(channels + num_classes), out_channels=[256, 128, 3],
                                          classifier=True, dim=1, width_multiplier=width_multiplier)
        self.regression = nn.Sequential(*layers)

    def forward(self, inputs):
        coords = inputs['coords']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2  # [B, C]

        features = self.features(coords)
        features = features.max(dim=-1, keepdim=False).values
        return self.regression(torch.cat([features, one_hot_vectors], dim=1))
