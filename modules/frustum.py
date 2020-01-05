import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.functional as PF

__all__ = ['FrustumPointNetLoss', 'get_box_corners_3d']


class FrustumPointNetLoss(nn.Module):
    def __init__(self, num_heading_angle_bins, num_size_templates, size_templates, box_loss_weight=1.0,
                 corners_loss_weight=10.0, heading_residual_loss_weight=20.0, size_residual_loss_weight=20.0):
        super().__init__()
        self.box_loss_weight = box_loss_weight
        self.corners_loss_weight = corners_loss_weight
        self.heading_residual_loss_weight = heading_residual_loss_weight
        self.size_residual_loss_weight = size_residual_loss_weight

        self.num_heading_angle_bins = num_heading_angle_bins
        self.num_size_templates = num_size_templates
        self.register_buffer('size_templates', size_templates.view(self.num_size_templates, 3))
        self.register_buffer(
            'heading_angle_bin_centers', torch.arange(0, 2 * np.pi, 2 * np.pi / self.num_heading_angle_bins)
        )

    def forward(self, inputs, targets):
        mask_logits = inputs['mask_logits']  # (B, 2, N)
        center_reg = inputs['center_reg']  # (B, 3)
        center = inputs['center']  # (B, 3)
        heading_scores = inputs['heading_scores']  # (B, NH)
        heading_residuals_normalized = inputs['heading_residuals_normalized']  # (B, NH)
        heading_residuals = inputs['heading_residuals']  # (B, NH)
        size_scores = inputs['size_scores']  # (B, NS)
        size_residuals_normalized = inputs['size_residuals_normalized']  # (B, NS, 3)
        size_residuals = inputs['size_residuals']  # (B, NS, 3)

        mask_logits_target = targets['mask_logits']  # (B, N)
        center_target = targets['center']  # (B, 3)
        heading_bin_id_target = targets['heading_bin_id']  # (B, )
        heading_residual_target = targets['heading_residual']  # (B, )
        size_template_id_target = targets['size_template_id']  # (B, )
        size_residual_target = targets['size_residual']  # (B, 3)

        batch_size = center.size(0)
        batch_id = torch.arange(batch_size, device=center.device)

        # Basic Classification and Regression losses
        mask_loss = F.cross_entropy(mask_logits, mask_logits_target)
        heading_loss = F.cross_entropy(heading_scores, heading_bin_id_target)
        size_loss = F.cross_entropy(size_scores, size_template_id_target)
        center_loss = PF.huber_loss(torch.norm(center_target - center, dim=-1), delta=2.0)
        center_reg_loss = PF.huber_loss(torch.norm(center_target - center_reg, dim=-1), delta=1.0)

        # Refinement losses for size/heading
        heading_residuals_normalized = heading_residuals_normalized[batch_id, heading_bin_id_target]  # (B, )
        heading_residual_normalized_target = heading_residual_target / (np.pi / self.num_heading_angle_bins)
        heading_residual_normalized_loss = PF.huber_loss(
            heading_residuals_normalized - heading_residual_normalized_target, delta=1.0
        )
        size_residuals_normalized = size_residuals_normalized[batch_id, size_template_id_target]  # (B, 3)
        size_residual_normalized_target = size_residual_target / self.size_templates[size_template_id_target]
        size_residual_normalized_loss = PF.huber_loss(
            torch.norm(size_residual_normalized_target - size_residuals_normalized, dim=-1), delta=1.0
        )

        # Bounding box losses
        heading = (heading_residuals[batch_id, heading_bin_id_target]
                   + self.heading_angle_bin_centers[heading_bin_id_target])  # (B, )
        # Warning: in origin code, size_residuals are added twice (issue #43 and #49 in charlesq34/frustum-pointnets)
        size = (size_residuals[batch_id, size_template_id_target]
                + self.size_templates[size_template_id_target])  # (B, 3)
        corners = get_box_corners_3d(centers=center, headings=heading, sizes=size, with_flip=False)  # (B, 3, 8)
        heading_target = self.heading_angle_bin_centers[heading_bin_id_target] + heading_residual_target  # (B, )
        size_target = self.size_templates[size_template_id_target] + size_residual_target  # (B, 3)
        corners_target, corners_target_flip = get_box_corners_3d(centers=center_target, headings=heading_target,
                                                                 sizes=size_target, with_flip=True)  # (B, 3, 8)
        corners_loss = PF.huber_loss(torch.min(
            torch.norm(corners - corners_target, dim=1), torch.norm(corners - corners_target_flip, dim=1)
        ), delta=1.0)
        # Summing up
        loss = mask_loss + self.box_loss_weight * (
                center_loss + center_reg_loss + heading_loss + size_loss
                + self.heading_residual_loss_weight * heading_residual_normalized_loss
                + self.size_residual_loss_weight * size_residual_normalized_loss
                + self.corners_loss_weight * corners_loss
        )

        return loss


def get_box_corners_3d(centers, headings, sizes, with_flip=False):
    """
    :param centers: coords of box centers, FloatTensor[N, 3]
    :param headings: heading angles, FloatTensor[N, ]
    :param sizes: box sizes, FloatTensor[N, 3]
    :param with_flip: bool, whether to return flipped box (headings + np.pi)
    :return:
        coords of box corners, FloatTensor[N, 3, 8]
        NOTE: corner points are in counter clockwise order, e.g.,
          2--1
        3--0 5
        7--4
    """
    l = sizes[:, 0]  # (N,)
    w = sizes[:, 1]  # (N,)
    h = sizes[:, 2]  # (N,)
    x_corners = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1)  # (N, 8)
    y_corners = torch.stack([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], dim=1)  # (N, 8)
    z_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1)  # (N, 8)

    c = torch.cos(headings)  # (N,)
    s = torch.sin(headings)  # (N,)
    o = torch.ones_like(headings)  # (N,)
    z = torch.zeros_like(headings)  # (N,)

    centers = centers.unsqueeze(-1)  # (B, 3, 1)
    corners = torch.stack([x_corners, y_corners, z_corners], dim=1)  # (N, 3, 8)
    R = torch.stack([c, z, s, z, o, z, -s, z, c], dim=1).view(-1, 3, 3)  # roty matrix: (N, 3, 3)
    if with_flip:
        R_flip = torch.stack([-c, z, -s, z, o, z, s, z, -c], dim=1).view(-1, 3, 3)
        return torch.matmul(R, corners) + centers, torch.matmul(R_flip, corners) + centers
    else:
        return torch.matmul(R, corners) + centers

    # centers = centers.unsqueeze(1)  # (B, 1, 3)
    # corners = torch.stack([x_corners, y_corners, z_corners], dim=-1)  # (N, 8, 3)
    # RT = torch.stack([c, z, -s, z, o, z, s, z, c], dim=1).view(-1, 3, 3)  # (N, 3, 3)
    # if with_flip:
    #     RT_flip = torch.stack([-c, z, s, z, o, z, -s, z, -c], dim=1).view(-1, 3, 3)  # (N, 3, 3)
    #     return torch.matmul(corners, RT) + centers, torch.matmul(corners, RT_flip) + centers  # (N, 8, 3)
    # else:
    #     return torch.matmul(corners, RT) + centers  # (N, 8, 3)

    # corners = torch.stack([x_corners, y_corners, z_corners], dim=1)  # (N, 3, 8)
    # R = torch.stack([c, z, s, z, o, z, -s, z, c], dim=1).view(-1, 3, 3)  # (N, 3, 3)
    # corners = torch.matmul(R, corners) + centers.unsqueeze(2)  # (N, 3, 8)
    # corners = corners.transpose(1, 2)  # (N, 8, 3)
