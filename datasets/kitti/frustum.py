import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from datasets.kitti.attributes import kitti_attributes as kitti
from utils.container import G


class FrustumKitti(dict):
    def __init__(self, root, num_points, split=None, classes=('Car', 'Pedestrian', 'Cyclist'),
                 num_heading_angle_bins=12, class_name_to_size_template_id=None,
                 from_rgb_detection=False, random_flip=False, random_shift=False, frustum_rotate=False):
        super().__init__()
        if class_name_to_size_template_id is None:
            class_name_to_size_template_id = {cat: cls for cls, cat in enumerate(kitti.class_names)}
        if not isinstance(split, (list, tuple)):
            if split is None:
                split = ['train', 'val']
            else:
                split = [split]
        if 'train' in split:
            self['train'] = _FrustumKittiDataset(
                root=root, num_points=num_points, split='train', classes=classes,
                num_heading_angle_bins=num_heading_angle_bins,
                class_name_to_size_template_id=class_name_to_size_template_id,
                random_flip=random_flip, random_shift=random_shift, frustum_rotate=frustum_rotate)
        if 'val' in split:
            self['val'] = _FrustumKittiDataset(
                root=root, num_points=num_points, split='val', classes=classes,
                num_heading_angle_bins=num_heading_angle_bins,
                class_name_to_size_template_id=class_name_to_size_template_id,
                random_flip=False, random_shift=False, frustum_rotate=frustum_rotate,
                from_rgb_detection=from_rgb_detection)


class _FrustumKittiDataset(Dataset):
    def __init__(self, root, num_points, split, classes, num_heading_angle_bins, class_name_to_size_template_id,
                 from_rgb_detection=False, random_flip=False, random_shift=False, frustum_rotate=False):
        """
        Frustum Kitti Dataset
        :param root: directory path to kitti prepared dataset
        :param num_points: number of points to process for each scene
        :param split: 'train' or 'test'
        :param classes: tuple of classes names
        :param num_heading_angle_bins: #heading angle bins, int
        :param class_name_to_size_template_id: dict
        :param from_rgb_detection: bool, if True we assume we do not have groundtruth, just return data elements.
        :param random_flip: bool, in 50% randomly flip the point cloud in left and right (after the frustum rotation)
        :param random_shift: bool, if True randomly shift the point cloud back and forth by a random distance
        :param frustum_rotate: bool, whether to do frustum rotation
        """
        assert split in ['train', 'val']
        self.root = root
        self.split = split
        self.classes = classes
        self.num_classes = len(classes)
        self.class_name_to_class_id = {cat: cls for cls, cat in enumerate(self.classes)}
        self.num_heading_angle_bins = num_heading_angle_bins
        self.class_name_to_size_template_id = class_name_to_size_template_id

        self.num_points = num_points
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.frustum_rotate = frustum_rotate
        self.from_rgb_detection = from_rgb_detection
        self.data = G()

        if self.from_rgb_detection:
            with open(os.path.join(self.root, f'frustum_carpedcyc_{split}_rgb_detection.pickle'), 'rb') as fp:
                self.data.ids = pickle.load(fp)
                self.data.boxes_2d = pickle.load(fp, encoding='latin1')
                self.data.point_clouds = pickle.load(fp, encoding='latin1')
                self.data.class_names = pickle.load(fp, encoding='latin1')
                # frustum_angle is clockwise angle from positive x-axis
                self.data.frustum_rotation_angles = pickle.load(fp, encoding='latin1')
                self.data.probs = pickle.load(fp, encoding='latin1')
        else:
            with open(os.path.join(self.root, f'frustum_carpedcyc_{split}.pickle'), 'rb') as fp:
                self.data.ids = pickle.load(fp)
                self.data.boxes_2d = pickle.load(fp, encoding='latin1')
                self.data.boxes_3d = pickle.load(fp, encoding='latin1')
                self.data.point_clouds = pickle.load(fp, encoding='latin1')
                self.data.mask_logits = pickle.load(fp, encoding='latin1')
                self.data.class_names = pickle.load(fp, encoding='latin1')
                self.data.heading_angles = pickle.load(fp, encoding='latin1')
                self.data.sizes = pickle.load(fp, encoding='latin1')
                # frustum_angle is clockwise angle from positive x-axis
                self.data.frustum_rotation_angles = pickle.load(fp, encoding='latin1')

    def __len__(self):
        return len(self.data.point_clouds)

    def __getitem__(self, index):
        # frustum rotation angle is from x clockwise to z
        # rotation angle is from z clockwise to x
        # frustum rotation angle shifted by pi/2 so that it can be directly used to adjust ground truth heading angle
        rotation_angle = np.pi / 2.0 + self.data.frustum_rotation_angles[index]

        # Compute one hot vector
        class_name = self.data.class_names[index]
        one_hot_vector = np.zeros(self.num_classes)
        one_hot_vector[self.class_name_to_class_id[class_name]] = 1
        one_hot_vector = one_hot_vector.astype(np.float32)

        # Get point cloud
        point_cloud = self.data.point_clouds[index]
        if self.frustum_rotate:
            # Use np.copy to avoid corrupting original data
            point_cloud = self.rotate_points_along_y(np.copy(point_cloud), rotation_angle)
        choice = np.random.choice(point_cloud.shape[0], self.num_points, replace=True)
        point_cloud = point_cloud[choice, :]

        if self.from_rgb_detection:
            return {'features': point_cloud.astype(np.float32).T, 'one_hot_vectors': one_hot_vector}, \
                   {'rotation_angle': rotation_angle.astype(np.float32), 'rgb_score': self.data.probs[index]}

        mask_logits = self.data.mask_logits[index][choice]
        center = (self.data.boxes_3d[index][0, :] + self.data.boxes_3d[index][6, :]) / 2.0
        heading_angle = self.data.heading_angles[index]
        size_template_id = self.class_name_to_size_template_id[class_name]
        size_residual = self.data.sizes[index] - kitti.class_name_to_size_template[class_name]
        if self.frustum_rotate:
            center = self.rotate_points_along_y(np.expand_dims(center, 0), rotation_angle).squeeze()
            heading_angle -= rotation_angle

        # Data Augmentation
        if self.random_flip:
            # note: rotation_angle won't be correct if we have random_flip so do not use it in case of random flipping
            if np.random.random() > 0.5:  # 50% chance flipping
                point_cloud[:, 0] = -point_cloud[:, 0]
                center[0] = -center[0]
                heading_angle = np.pi - heading_angle
        if self.random_shift:
            dist = np.sqrt(np.sum(center[0] ** 2 + center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_cloud[:, 2] += shift
            center[2] += shift

        heading_bin_id, heading_residual = self.angle_to_bin_id(heading_angle, self.num_heading_angle_bins)

        return {'features': point_cloud.astype(np.float32).T, 'one_hot_vectors': one_hot_vector},\
               {'mask_logits': mask_logits.astype(np.int64), 'center': center.astype(np.float32),
                'heading_bin_id': heading_bin_id,  'heading_residual': np.array(heading_residual, dtype=np.float32),
                'size_template_id': size_template_id, 'size_residual': size_residual.astype(np.float32),
                'class_id': self.class_name_to_class_id[class_name]}

    @staticmethod
    def rotate_points_along_y(features, rotation_angle):
        """
        (https://github.com/charlesq34/frustum-pointnets/blob/master/sunrgbd/sunrgbd_detection/roi_seg_box3d_dataset.py)
        :param features: numpy array (N,C), first 3 channels are XYZ-coords
                         z is facing forward, x is left ward, y is downward
        :param rotation_angle: float, from z to x axis, unit: rad (rotate axis from z to x = rotate coords from x to z)
        :return:
            features: numpy array (N, C) with [0, 2] rotated
        """
        v_cos = np.cos(rotation_angle)
        v_sin = np.sin(rotation_angle)
        # rotation_matrix = np.array([[v_cos, -v_sin], [v_sin, v_cos]])
        rotation_matrix_transpose = [[v_cos, v_sin], [-v_sin, v_cos]]
        features[:, [0, 2]] = np.dot(features[:, [0, 2]], rotation_matrix_transpose)
        return features

    @staticmethod
    def angle_to_bin_id(angle, num_angle_bins):
        """
        (https://github.com/charlesq34/frustum-pointnets/blob/master/sunrgbd/sunrgbd_detection/roi_seg_box3d_dataset.py)
        Convert continuous angle to discrete bin and residual.
        :param angle: float, unit: rad
        :param num_angle_bins: int, #angle bins
        :return:
            bin_id: int, bin id
            angle_residual: float, bin_id * (2pi/N) + angle_residual = angle
        """
        angle = angle % (2 * np.pi)
        assert 0 <= angle <= 2 * np.pi
        angle_per_bin = 2 * np.pi / float(num_angle_bins)
        shifted_angle = (angle + angle_per_bin / 2) % (2 * np.pi)
        bin_id = int(shifted_angle / angle_per_bin)
        angle_residual = shifted_angle - (bin_id * angle_per_bin + angle_per_bin / 2)
        return bin_id, angle_residual
