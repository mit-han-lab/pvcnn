from torch.autograd import Function

from modules.functional.backend import _backend

__all__ = ['nearest_neighbor_interpolate']


class NeighborInterpolation(Function):
    @staticmethod
    def forward(ctx, points_coords, centers_coords, centers_features):
        """
        :param ctx:
        :param points_coords: coordinates of points, FloatTensor[B, 3, N]
        :param centers_coords: coordinates of centers, FloatTensor[B, 3, M]
        :param centers_features: features of centers, FloatTensor[B, C, M]
        :return:
            points_features: features of points, FloatTensor[B, C, N]
        """
        centers_coords = centers_coords.contiguous()
        points_coords = points_coords.contiguous()
        centers_features = centers_features.contiguous()
        points_features, indices, weights = _backend.three_nearest_neighbors_interpolate_forward(
            points_coords, centers_coords, centers_features
        )
        ctx.save_for_backward(indices, weights)
        ctx.num_centers = centers_coords.size(-1)
        return points_features

    @staticmethod
    def backward(ctx, grad_output):
        indices, weights = ctx.saved_tensors
        grad_centers_features = _backend.three_nearest_neighbors_interpolate_backward(
            grad_output.contiguous(), indices, weights, ctx.num_centers
        )
        return None, None, grad_centers_features


nearest_neighbor_interpolate = NeighborInterpolation.apply
