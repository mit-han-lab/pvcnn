import torch
import torch.nn as nn

from modules import SharedMLP, PVConv

__all__ = ['PVCNN']


class PVCNN(nn.Module):
    blocks = ((64, 1, 32), (64, 2, 16), (128, 1, 16), 1024)

    def __init__(self, num_classes, in_channels=9, width_multiplier=1, voxel_resolution_multiplier=1, **kwargs):
        super().__init__()
        r, vr = width_multiplier, voxel_resolution_multiplier

        layers, concat_channels = [], 0
        for out_channels, num_blocks, voxel_resolution in self.blocks[:-1]:
            out_channels, voxel_resolution = int(r * out_channels), int(vr * voxel_resolution)
            for _ in range(num_blocks):
                layers.append(PVConv(in_channels, out_channels, 3, resolution=voxel_resolution, with_se=False))
                in_channels = out_channels
                concat_channels += out_channels
        out_channels = int(min(1, r) * self.blocks[-1])
        concat_channels += out_channels
        layers.append(SharedMLP(in_channels, out_channels))
        self.point_features = nn.ModuleList(layers)
        
        self.cloud_features = nn.Sequential(
            nn.Linear(out_channels, int(r * 256)),
            nn.BatchNorm1d(int(r * 256)),
            nn.ReLU(inplace=True),
            nn.Linear(int(r * 256), int(r * 128)),
            nn.BatchNorm1d(int(r * 128)),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            SharedMLP(in_channels=concat_channels + int(r * 128), out_channels=int(r * 512)),
            nn.Dropout(0.3),
            SharedMLP(in_channels=int(r * 512), out_channels=int(r * 256)),
            nn.Dropout(0.3),
            nn.Conv1d(int(r * 256), num_classes, 1)
        )

    def forward(self, inputs):
        coords = inputs[:, :3, :]
        features_list = []
        for i in range(len(self.point_features)):
            inputs, _ = self.point_features[i]((inputs, coords))
            features_list.append(inputs)
        # inputs: num_batches * 1024 * num_points -> num_batches * 1024 -> num_batches * 128
        inputs = self.cloud_features(inputs.max(dim=-1, keepdim=False).values).unsqueeze(-1).repeat([1, 1, coords.size(-1)])
        features_list.append(inputs)
        inputs = torch.cat(features_list, dim=1)
        return self.classifier(inputs)
