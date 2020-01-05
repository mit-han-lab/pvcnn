import torch
import torch.nn as nn

from models.utils import create_pointnet_components, create_mlp_components

__all__ = ['PointNet']


class PointNet(nn.Module):
    blocks = ((64, 3, None), (128, 1, None), (1024, 1, None))

    def __init__(self, num_classes, extra_feature_channels=6, width_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3

        layers, channels_point, _ = create_pointnet_components(blocks=self.blocks, in_channels=self.in_channels,
                                                               width_multiplier=width_multiplier)
        self.point_features = nn.Sequential(*layers)

        layers, channels_cloud = create_mlp_components(in_channels=channels_point, out_channels=[256, 128],
                                                       classifier=False, dim=1, width_multiplier=width_multiplier)
        self.cloud_features = nn.Sequential(*layers)

        layers, _ = create_mlp_components(
            in_channels=(channels_point + channels_cloud), out_channels=[512, 256, 0.3, num_classes],
            classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['features']

        point_features = self.point_features(inputs)
        cloud_features = self.cloud_features(point_features.max(dim=-1, keepdim=False).values)
        features = torch.cat([point_features, cloud_features.unsqueeze(-1).repeat([1, 1, inputs.size(-1)])], dim=1)
        return self.classifier(features)
