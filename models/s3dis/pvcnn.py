import torch
import torch.nn as nn

from models.utils import create_mlp_components, create_pointnet_components

__all__ = ['PVCNN']


class PVCNN(nn.Module):
    blocks = ((64, 1, 32), (64, 2, 16), (128, 1, 16), (1024, 1, None))

    def __init__(self, num_classes, extra_feature_channels=6, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)

        layers, channels_cloud = create_mlp_components(
            in_channels=channels_point, out_channels=[256, 128],
            classifier=False, dim=1, width_multiplier=width_multiplier)
        self.cloud_features = nn.Sequential(*layers)

        layers, _ = create_mlp_components(
            in_channels=(concat_channels_point + channels_cloud),
            out_channels=[512, 0.3, 256, 0.3, num_classes],
            classifier=True, dim=2, width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['features']

        coords = inputs[:, :3, :]
        out_features_list = []
        for i in range(len(self.point_features)):
            inputs, _ = self.point_features[i]((inputs, coords))
            out_features_list.append(inputs)
        # inputs: num_batches * 1024 * num_points -> num_batches * 1024 -> num_batches * 128
        inputs = self.cloud_features(inputs.max(dim=-1, keepdim=False).values)
        out_features_list.append(inputs.unsqueeze(-1).repeat([1, 1, coords.size(-1)]))
        return self.classifier(torch.cat(out_features_list, dim=1))
