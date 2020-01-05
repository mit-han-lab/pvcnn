import torch
import torch.nn as nn

from models.utils import create_pointnet2_sa_components, create_mlp_components

__all__ = ['BoxEstimationPointNet2']


class BoxEstimationNet2(nn.Module):
    def __init__(self, num_classes, sa_blocks, num_heading_angle_bins, num_size_templates,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = 3
        self.num_classes = num_classes

        sa_layers, _, channels_sa_features, num_centers = create_pointnet2_sa_components(
            sa_blocks=sa_blocks, extra_feature_channels=0, with_se=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.features = nn.Sequential(*sa_layers)

        layers, _ = create_mlp_components(
            in_channels=(channels_sa_features * num_centers + num_classes),
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

        features, _ = self.features((None, coords))
        features = features.view(features.size(0), -1)
        return self.classifier(torch.cat([features, one_hot_vectors], dim=1))


class BoxEstimationPointNet2(BoxEstimationNet2):
    sa_blocks = [
        (None, (128, 0.2, 64, (64, 64, 128))),
        (None, (32, 0.4, 64, (128, 128, 256))),
        (None, (None, None, None, (256, 256, 512))),
    ]

    def __init__(self, num_classes=3, num_heading_angle_bins=12, num_size_templates=8, width_multiplier=1):
        super().__init__(num_classes=num_classes, sa_blocks=self.sa_blocks,
                         num_heading_angle_bins=num_heading_angle_bins, num_size_templates=num_size_templates,
                         width_multiplier=width_multiplier)
