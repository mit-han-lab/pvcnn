import torch
import torch.nn as nn

from models.utils import create_pointnet2_sa_components, create_pointnet2_fp_modules, create_mlp_components

__all__ = ['InstanceSegmentationPointNet2']


class InstanceSegmentationNet2(nn.Module):
    def __init__(self, num_classes, sa_blocks, fp_blocks, extra_feature_channels,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3
        self.num_classes = num_classes

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=sa_blocks, extra_feature_channels=extra_feature_channels, with_se=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        # use one hot vector in the first fp module
        sa_in_channels[-1] += num_classes
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(
            in_channels=channels_fp_features, out_channels=[128, 0.3, 2],
            classifier=True, dim=2, width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        features = inputs['features']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2  # [B, C]

        coords, extra_features = features[:, :3, :].contiguous(), features[:, 3:, :].contiguous()
        coords_list, in_features_list = [], []
        for sa_module in self.sa_layers:
            in_features_list.append(extra_features)
            coords_list.append(coords)
            extra_features, coords = sa_module((extra_features, coords))
        in_features_list[0] = features.contiguous()

        features = torch.cat(
            [extra_features, one_hot_vectors.unsqueeze(-1).repeat([1, 1, extra_features.size(-1)])], dim=1
        )
        for fp_idx, fp_module in enumerate(self.fp_layers):
            features, coords = fp_module(
                (coords_list[-1 - fp_idx], coords, features, in_features_list[-1 - fp_idx])
            )
        return self.classifier(features)


class InstanceSegmentationPointNet2(InstanceSegmentationNet2):
    sa_blocks = [
        (None, (128, [0.2, 0.4, 0.8], [32, 64, 128], [(32, 32, 64), (64, 64, 128), (64, 96, 128)])),
        (None, (32, [0.4, 0.8, 1.6], [64, 64, 128], [(64, 64, 128), (128, 128, 256), (128, 128, 256)])),
        (None, (None, None, None, (128, 256, 1024))),
    ]
    fp_blocks = [((128, 128), None), ((128, 128), None), ((128, 128), None)]

    def __init__(self, num_classes=3, extra_feature_channels=1, width_multiplier=1):
        super().__init__(
            num_classes=num_classes, sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks,
            extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier
        )
