import numpy as np
import torch

from modules.frustum import get_box_corners_3d
from meters.kitti.utils import get_box_iou_3d

__all__ = ['MeterFrustumKitti']


class MeterFrustumKitti:
    def __init__(self, num_heading_angle_bins, num_size_templates, size_templates, class_name_to_class_id,
                 metric='iou_3d'):
        super().__init__()
        assert metric in ['iou_2d', 'iou_3d', 'accuracy', 'iou_3d_accuracy', 'iou_3d_class_accuracy']
        self.metric = metric
        self.num_heading_angle_bins = num_heading_angle_bins
        self.num_size_templates = num_size_templates
        self.size_templates = size_templates.view(self.num_size_templates, 3)
        self.heading_angle_bin_centers = torch.arange(0, 2 * np.pi, 2 * np.pi / self.num_heading_angle_bins)
        self.class_name_to_class_id = class_name_to_class_id
        self.reset()

    def reset(self):
        self.total_seen_num = 0
        self.total_correct_num = 0
        self.iou_3d_corrent_num = 0
        self.iou_2d_sum = 0
        self.iou_3d_sum = 0
        self.iou_3d_corrent_num_per_class = {cls: 0 for cls in self.class_name_to_class_id.keys()}
        self.total_seen_num_per_class = {cls: 0 for cls in self.class_name_to_class_id.keys()}

    def update(self, outputs, targets):
        if self.metric == 'accuracy':
            mask_logits = outputs['mask_logits']
            mask_logits_target = targets['mask_logits']
            self.total_seen_num += mask_logits_target.numel()
            self.total_correct_num += torch.sum(mask_logits.argmax(dim=1) == mask_logits_target).item()
        else:
            center = outputs['center']  # (B, 3)
            heading_scores = outputs['heading_scores']  # (B, NH)
            heading_residuals = outputs['heading_residuals']  # (B, NH)
            size_scores = outputs['size_scores']  # (B, NS)
            size_residuals = outputs['size_residuals']  # (B, NS, 3)

            center_target = targets['center']  # (B, 3)
            heading_bin_id_target = targets['heading_bin_id']  # (B, )
            heading_residual_target = targets['heading_residual']  # (B, )
            size_template_id_target = targets['size_template_id']  # (B, )
            size_residual_target = targets['size_residual']  # (B, 3)
            class_id_target = targets['class_id'].cpu().numpy()  # (B, )

            batch_size = center.size(0)
            batch_id = torch.arange(batch_size, device=center.device)
            self.size_templates = self.size_templates.to(center.device)
            self.heading_angle_bin_centers = self.heading_angle_bin_centers.to(center.device)

            heading_bin_id = torch.argmax(heading_scores, dim=1)
            heading = self.heading_angle_bin_centers[heading_bin_id] + heading_residuals[batch_id, heading_bin_id]
            size_template_id = torch.argmax(size_scores, dim=1)
            size = self.size_templates[size_template_id] + size_residuals[batch_id, size_template_id]  # (B, 3)
            corners = get_box_corners_3d(centers=center, headings=heading, sizes=size, with_flip=False)  # (B, 8, 3)
            heading_target = self.heading_angle_bin_centers[heading_bin_id_target] + heading_residual_target  # (B, )
            size_target = self.size_templates[size_template_id_target] + size_residual_target  # (B, 3)
            corners_target = get_box_corners_3d(centers=center_target, headings=heading_target,
                                                sizes=size_target, with_flip=False)  # (B, 8, 3)
            iou_3d, iou_2d = get_box_iou_3d(corners.cpu().numpy(), corners_target.cpu().numpy())
            self.iou_2d_sum += iou_2d.sum()
            self.iou_3d_sum += iou_3d.sum()
            self.iou_3d_corrent_num += np.sum(iou_3d >= 0.7)
            self.total_seen_num += batch_size
            for cls, cls_id in self.class_name_to_class_id.items():
                mask = (class_id_target == cls_id)
                self.iou_3d_corrent_num_per_class[cls] += np.sum(iou_3d[mask] >= (0.7 if cls == 'Car' else 0.5))
                self.total_seen_num_per_class[cls] += np.sum(mask)

    def compute(self):
        if self.metric == 'iou_3d':
            return self.iou_3d_sum / self.total_seen_num
        elif self.metric == 'iou_2d':
            return self.iou_2d_sum / self.total_seen_num
        elif self.metric == 'accuracy':
            return self.total_correct_num / self.total_seen_num
        elif self.metric == 'iou_3d_accuracy':
            return self.iou_3d_corrent_num / self.total_seen_num
        elif self.metric == 'iou_3d_class_accuracy':
            return sum(self.iou_3d_corrent_num_per_class[cls] / max(self.total_seen_num_per_class[cls], 1)
                       for cls in self.class_name_to_class_id.keys()) / len(self.class_name_to_class_id)
        else:
            raise KeyError
