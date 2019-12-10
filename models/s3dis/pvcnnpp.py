import torch.nn as nn

from modules import PVConv, PointNetSAModule, PointNetFPModule, SharedMLP

__all__ = ['PVCNN2']


class PVCNN2(nn.Module):
    sa_blocks = [
        (32, 2, 32, (1024, 0.1, 32, (32, 64))),
        (64, 3, 16, (256, 0.2, 32, (64, 128))),
        (128, 3, 8, (64, 0.4, 32, (128, 256))),
        (None, None, None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), 256, 1, 8),
        ((256, 256), 256, 1, 8),
        ((256, 128), 128, 2, 16),
        ((128, 128, 64), 64, 1, 32),
    ]

    def __init__(self, num_classes, in_channels=9, width_multiplier=1, voxel_resolution_multiplier=1, **kwargs):
        super(PVCNN2, self).__init__()
        r, vr = width_multiplier, voxel_resolution_multiplier

        sa_layers, sa_out_channels = [], [in_channels - 3]
        for out_channels, num_blocks, voxel_resolution, sa_configs in self.sa_blocks:
            sa_blocks = []
            if out_channels is not None:
                out_channels, voxel_resolution = int(r * out_channels), int(vr * voxel_resolution)
                for _ in range(num_blocks):
                    sa_blocks.append(PVConv(in_channels, out_channels, 3, resolution=voxel_resolution, with_se=True))
                    in_channels = out_channels
            num_centers, radius, num_neighbors, out_channels = sa_configs
            out_channels = tuple(r * oc for oc in out_channels)
            sa_blocks.append(
                PointNetSAModule(num_centers=num_centers, radius=radius, num_neighbors=num_neighbors,
                                 in_channels=in_channels, out_channels=out_channels, include_coordinates=True)
            )
            in_channels = out_channels[-1]
            sa_out_channels.append(in_channels)
            if len(sa_blocks) == 1:
                sa_layers.append(sa_blocks[0])
            else:
                sa_layers.append(nn.Sequential(*sa_blocks))
        self.sa_layers = nn.ModuleList(sa_layers)

        sa_out_channels = sa_out_channels[:-1]
        fp_layers = []
        for fp_idx, (fp_configs, out_channels, num_blocks, voxel_resolution) in enumerate(self.fp_blocks):
            fp_blocks = []
            fp_configs = tuple(r * oc for oc in fp_configs)
            fp_blocks.append(
                PointNetFPModule(in_channels=in_channels + sa_out_channels[-1-fp_idx], out_channels=fp_configs)
            )
            in_channels = fp_configs[-1]
            out_channels, voxel_resolution = int(r * out_channels), int(vr * voxel_resolution)
            for _ in range(num_blocks):
                fp_blocks.append(PVConv(in_channels, out_channels, 3, resolution=voxel_resolution, with_se=True))
                in_channels = out_channels
            if len(fp_blocks) == 1:
                fp_layers.append(fp_blocks[0])
            else:
                fp_layers.append(nn.Sequential(*fp_blocks))
        self.fp_layers = nn.ModuleList(fp_layers)

        self.classifier = nn.Sequential(
            SharedMLP(in_channels=in_channels, out_channels=int(r * 128)),
            nn.Dropout(0.5),
            nn.Conv1d(int(r * 128), num_classes, 1)
        )

    def forward(self, inputs):
        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, features_list = [], []
        for sa_blocks in self.sa_layers:
            features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
        features_list[0] = inputs[:, 3:, :].contiguous()

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1-fp_idx], coords, features, features_list[-1-fp_idx]))

        return self.classifier(features)
