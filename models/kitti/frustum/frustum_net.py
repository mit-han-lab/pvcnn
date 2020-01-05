import functools

import numpy as np
import torch.nn as nn

import modules.functional as F
from models.kitti.frustum.box_estimation import *
from models.kitti.frustum.segmentation import *
from models.kitti.frustum.center_regression_net import CenterRegressionNet

__all__ = ['FrustumPointNet', 'FrustumPointNet2', 'FrustumPVCNNE']


class FrustumNet(nn.Module):
    def __init__(self, num_classes, instance_segmentation_net, box_estimation_net,
                 num_heading_angle_bins, num_size_templates, num_points_per_object,
                 size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__()
        if not isinstance(width_multiplier, (list, tuple)):
            width_multiplier = [width_multiplier] * 3
        self.in_channels = 3 + extra_feature_channels
        self.num_classes = num_classes
        self.num_heading_angle_bins = num_heading_angle_bins
        self.num_size_templates = num_size_templates
        self.num_points_per_object = num_points_per_object

        self.inst_seg_net = instance_segmentation_net(num_classes=num_classes,
                                                      extra_feature_channels=extra_feature_channels,
                                                      width_multiplier=width_multiplier[0])
        self.center_reg_net = CenterRegressionNet(num_classes=num_classes, width_multiplier=width_multiplier[1])
        self.box_est_net = box_estimation_net(num_classes=num_classes, num_heading_angle_bins=num_heading_angle_bins,
                                              num_size_templates=num_size_templates,
                                              width_multiplier=width_multiplier[2])
        self.register_buffer('size_templates', size_templates.view(1, self.num_size_templates, 3))

    def forward(self, inputs):
        features = inputs['features']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2

        # foreground/background segmentation
        mask_logits = self.inst_seg_net({'features': features, 'one_hot_vectors': one_hot_vectors})
        # mask out Background points
        foreground_coords, foreground_coords_mean, _ = F.logits_mask(
            coords=features[:, :3, :], logits=mask_logits, num_points_per_object=self.num_points_per_object
        )
        # center regression
        delta_coords = self.center_reg_net({'coords': foreground_coords, 'one_hot_vectors': one_hot_vectors})
        foreground_coords = foreground_coords - delta_coords.unsqueeze(-1)
        # box estimation
        estimation = self.box_est_net({'coords': foreground_coords, 'one_hot_vectors': one_hot_vectors})
        estimations = estimation.split([3, self.num_heading_angle_bins, self.num_heading_angle_bins,
                                        self.num_size_templates, self.num_size_templates * 3], dim=-1)

        # parse results
        outputs = dict()
        outputs['mask_logits'] = mask_logits
        outputs['center_reg'] = foreground_coords_mean + delta_coords
        outputs['center'] = estimations[0] + outputs['center_reg']
        outputs['heading_scores'] = estimations[1]
        outputs['heading_residuals_normalized'] = estimations[2]
        outputs['heading_residuals'] = estimations[2] * (np.pi / self.num_heading_angle_bins)
        outputs['size_scores'] = estimations[3]
        size_residuals_normalized = estimations[4].view(-1, self.num_size_templates, 3)
        outputs['size_residuals_normalized'] = size_residuals_normalized
        outputs['size_residuals'] = size_residuals_normalized * self.size_templates

        return outputs


class FrustumPointNet(FrustumNet):
    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object,
                 size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__(num_classes=num_classes, instance_segmentation_net=InstanceSegmentationPointNet,
                         box_estimation_net=BoxEstimationPointNet, num_heading_angle_bins=num_heading_angle_bins,
                         num_size_templates=num_size_templates, num_points_per_object=num_points_per_object,
                         size_templates=size_templates, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier)


class FrustumPointNet2(FrustumNet):
    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object,
                 size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__(num_classes=num_classes, instance_segmentation_net=InstanceSegmentationPointNet2,
                         box_estimation_net=BoxEstimationPointNet2, num_heading_angle_bins=num_heading_angle_bins,
                         num_size_templates=num_size_templates, num_points_per_object=num_points_per_object,
                         size_templates=size_templates, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier)


class FrustumPVCNNE(FrustumNet):
    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object,
                 size_templates, extra_feature_channels=1, width_multiplier=1, voxel_resolution_multiplier=1):
        instance_segmentation_net = functools.partial(InstanceSegmentationPVCNN,
                                                      voxel_resolution_multiplier=voxel_resolution_multiplier)
        super().__init__(num_classes=num_classes, instance_segmentation_net=instance_segmentation_net,
                         box_estimation_net=BoxEstimationPointNet, num_heading_angle_bins=num_heading_angle_bins,
                         num_size_templates=num_size_templates, num_points_per_object=num_points_per_object,
                         size_templates=size_templates, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier)
