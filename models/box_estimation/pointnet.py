import torch
import torch.nn as nn

from models.utils import create_pointnet_components, create_mlp_components

__all__ = ['BoxEstimationPointNet']


class BoxEstimationNet(nn.Module):
    def __init__(self, num_classes, blocks, num_heading_angle_bins, num_size_templates,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = 3
        self.num_classes = num_classes

        layers, channels_point, _ = create_pointnet_components(
            blocks=blocks, in_channels=self.in_channels, with_se=False, normalize=True, eps=1e-15,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.features = nn.Sequential(*layers)

        layers, _ = create_mlp_components(
            in_channels=channels_point + num_classes,
            out_channels=[512, 256, (3 + num_heading_angle_bins * 2 + num_size_templates * 4)],
            classifier=True, dim=1, width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)
        # outputs are: center(x, y, z)
        #              + num_heading_angle_bins * (score, delta angle)
        #              + num_size_templates * (score, delta x, delta y, delta z)

    def forward(self, inputs):
        coords = inputs['coords']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2  # [B, C]

        features, _ = self.features((coords, coords))
        features = features.max(dim=-1, keepdim=False).values
        return self.classifier(torch.cat([features, one_hot_vectors], dim=1))


class BoxEstimationPointNet(BoxEstimationNet):
    blocks = ((128, 2, None), (256, 1, None), (512, 1, None))

    def __init__(self, num_classes=3, num_heading_angle_bins=12, num_size_templates=8, width_multiplier=1):
        super().__init__(num_classes=num_classes, blocks=self.blocks, num_heading_angle_bins=num_heading_angle_bins,
                         num_size_templates=num_size_templates, width_multiplier=width_multiplier)
