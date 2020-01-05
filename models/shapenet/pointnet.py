import torch
import torch.nn as nn

from modules import SharedMLP

__all__ = ['PointNet']


class Transformer(nn.Module):
    def __init__(self, channels):
        super(Transformer, self).__init__()
        self.channels = channels

        self.features = nn.Sequential(
            SharedMLP(self.channels, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 1024),
        )
        self.tranformer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.channels * self.channels)
        )

    def forward(self, inputs):
        transform_weight = self.tranformer(torch.max(self.features(inputs), dim=-1, keepdim=False).values)
        transform_weight = transform_weight.view(-1, self.channels, self.channels)
        transform_weight = transform_weight + torch.eye(self.channels, device=transform_weight.device)
        outputs = torch.bmm(transform_weight, inputs)
        return outputs


class PointNet(nn.Module):
    blocks = ((True, 64, 1), (False, 128, 2), (True, 512, 1), (False, 2048, 1))

    def __init__(self, num_classes, num_shapes, with_transformer=False, extra_feature_channels=0, width_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        r = width_multiplier
        self.in_channels = in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        self.with_transformer = with_transformer

        layers, concat_channels = [], 0
        for with_transformer_before, out_channels, num_blocks in self.blocks:
            with_transformer_before = with_transformer_before and with_transformer
            out_channels = int(r * out_channels)
            for block_index in range(num_blocks):
                if with_transformer_before and block_index == 0:
                    layers.append(nn.Sequential(Transformer(in_channels), SharedMLP(in_channels, out_channels)))
                else:
                    layers.append(SharedMLP(in_channels, out_channels))
                in_channels = out_channels
                concat_channels += out_channels
        self.point_features = nn.ModuleList(layers)

        self.classifier = nn.Sequential(
            SharedMLP(in_channels=in_channels + concat_channels + num_shapes, out_channels=int(r * 256)),
            nn.Dropout(0.2),
            SharedMLP(in_channels=int(r * 256), out_channels=int(r * 256)),
            nn.Dropout(0.2),
            SharedMLP(in_channels=int(r * 256), out_channels=int(r * 128)),
            nn.Conv1d(int(r * 128), num_classes, 1)
        )

    def forward(self, inputs):
        # inputs: [B, in_channels + S, N]
        assert inputs.size(1) == self.in_channels + self.num_shapes
        features = inputs[:, :self.in_channels, :]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)

        out_features_list = [one_hot_vectors]
        for i in range(len(self.point_features)):
            features = self.point_features[i](features)
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        return self.classifier(torch.cat(out_features_list, dim=1))
