import torch
import torch.nn as nn

from models.utils import create_pointnet_components, create_mlp_components

__all__ = ['InstanceSegmentationPointNet', 'InstanceSegmentationPVCNN']


class InstanceSegmentationNet(nn.Module):
    def __init__(self, num_classes, point_blocks, cloud_blocks, extra_feature_channels,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3
        self.num_classes = num_classes

        layers, channels_point, _ = create_pointnet_components(
            blocks=point_blocks, in_channels=self.in_channels, with_se=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.Sequential(*layers)

        layers, channels_cloud, _ = create_pointnet_components(
            blocks=cloud_blocks, in_channels=channels_point, with_se=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.cloud_features = nn.Sequential(*layers)

        layers, _ = create_mlp_components(
            in_channels=(channels_point + channels_cloud + num_classes), out_channels=[512, 256, 128, 128, 0.5, 2],
            classifier=True, dim=2, width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        features = inputs['features']
        num_points = features.size(-1)
        one_hot_vectors = inputs['one_hot_vectors'].unsqueeze(-1).repeat([1, 1, num_points])
        assert one_hot_vectors.dim() == 3  # [B, C, N]

        point_features, point_coords = self.point_features((features, features[:, :3, :]))
        cloud_features, _ = self.cloud_features((point_features, point_coords))
        cloud_features = cloud_features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points])
        return self.classifier(torch.cat([one_hot_vectors, point_features, cloud_features], dim=1))


class InstanceSegmentationPointNet(InstanceSegmentationNet):
    point_blocks = ((64, 3, None),)
    cloud_blocks = ((128, 1, None), (1024, 1, None))

    def __init__(self, num_classes=3, extra_feature_channels=1, width_multiplier=1):
        super().__init__(
            num_classes=num_classes, point_blocks=self.point_blocks, cloud_blocks=self.cloud_blocks,
            extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier
        )


class InstanceSegmentationPVCNN(InstanceSegmentationNet):
    point_blocks = ((64, 2, 16), (64, 1, 12), (128, 1, 12), (1024, 1, None))
    cloud_blocks = ()

    def __init__(self, num_classes=3, extra_feature_channels=1, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, point_blocks=self.point_blocks, cloud_blocks=self.cloud_blocks,
            extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
