import torch
import torch.nn as nn

import modules.functional as F
from modules.ball_query import BallQuery
from modules.shared_mlp import SharedMLP

__all__ = ['PointNetSAModule', 'PointNetFPModule']


class PointNetSAModule(nn.Module):
    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        if not isinstance(radius, (list, tuple)):
            radius = [radius]
        if not isinstance(num_neighbors, (list, tuple)):
            num_neighbors = [num_neighbors] * len(radius)
        assert len(radius) == len(num_neighbors)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]] * len(radius)
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels] * len(radius)
        assert len(radius) == len(out_channels)

        groupers = nn.ModuleList()
        mlps = nn.ModuleList()
        for _radius, _out_channels, _num_neighbors in zip(radius, out_channels, num_neighbors):
            groupers.append(
                BallQuery(radius=_radius, num_neighbors=_num_neighbors, include_coordinates=include_coordinates)
            )
            mlps.append(
                SharedMLP(in_channels=in_channels + (3 if include_coordinates else 0),
                          out_channels=_out_channels, dim=2)
            )

        self.num_centers = num_centers
        self.groupers = groupers
        self.mlps = mlps

    def forward(self, inputs):
        features, coords = inputs
        centers_coords = F.furthest_point_sampling(coords, self.num_centers)
        features_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            neighbor_features = grouper(coords, centers_coords, features)
            neighbor_features = mlp(neighbor_features)
            neighbor_features = neighbor_features.max(dim=-1).values
            features_list.append(neighbor_features)
        return torch.cat(features_list, dim=1), centers_coords

    def extra_repr(self):
        return 'num_centers={}'.format(self.num_centers)


class PointNetFPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP(in_channels=in_channels, out_channels=out_channels, dim=1)

    def forward(self, inputs):
        if len(inputs) == 3:
            points_coords, centers_coords, centers_features = inputs
            points_features = None
        else:
            points_coords, centers_coords, centers_features, points_features = inputs
        interpolated_features = F.nearest_neighbor_interpolate(points_coords, centers_coords, centers_features)
        if points_features is not None:
            interpolated_features = torch.cat(
                [interpolated_features, points_features], dim=1
            )
        return self.mlp(interpolated_features), points_coords
