from torch.autograd import Function

from modules.functional.backend import _backend

__all__ = ['furthest_point_sampling']


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, coords, num_samples):
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance to the sampled point set
        :param ctx:
        :param coords: coordinates of points, FloatTensor[B, 3, N]
        :param num_samples: int, M
        :return:
            centers_coords: coordinates of sampled centers, FloatTensor[B, 3, M]
        """
        coords = coords.contiguous()
        indices = _backend.furthest_point_sampling(coords, num_samples)
        ctx.save_for_backward(indices)
        ctx.num_points = coords.size(-1)
        return _backend.gather_features_forward(coords, indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors
        grad_coords = _backend.gather_features_backward(grad_output.contiguous(), indices, ctx.num_points)
        return grad_coords, None


furthest_point_sampling = FurthestPointSampling.apply
