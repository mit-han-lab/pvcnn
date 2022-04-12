import json
import os

import numpy as np
from torch.utils.data import Dataset

__all__ = ['Custom']

class Custom_Dataset(Dataset):
    def __init__(self, root, num_points, split='train', with_normal=True, normalize=True, jitter=True, data_aug=True):
        
        assert split in ['train', 'test']
    
        self.dataset_path = root
        self.num_points = num_points
        self.split = split
        self.with_normal = with_normal  # add rgb features or not
        self.normalize = normalize
        self.jitter = jitter
        self.data_aug = data_aug        

        self.file_paths, self.file_labels = [], []
                    
        for dirPath, dirNames, fileNames in os.walk(os.path.join(self.dataset_path, split)):
            if dirPath.split("\\")[-1] == "GT":
                for f in fileNames:
                    self.file_labels.append(os.path.join(dirPath, f))
            else:
                for f in fileNames:
                    self.file_paths.append(os.path.join(dirPath, f))
      
        self.cache = {}  # from index to (point_set, rgb, gt) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        
        if index in self.cache:
            coords, normal, label = self.cache[index]
        else:
            file_path = self.file_paths[index]
            label_path = self.file_labels[index]

            data = np.load(file_path).astype(np.float32)  # load point cloud data (.npy file)
            
            coords = data[:, :3]
            
            normal = data[:, 3:6]
            label = np.load(label_path).astype(np.int64) # load GT [classes, num_pc]            

            if len(self.cache) < self.cache_size:
                self.cache[index] = (coords, normal, label)
            
        ''' Do Data Augmentation '''
        if self.data_aug:
            if np.random.random() < 0.5:  # 50% of all training data to do data augmentation
                if np.random.random() < 0.5: # 50% chance to do x-horizontal flip
                    coords[:, 0] = -coords[:, 0]
                else:
                    coords[:, 1] = -coords[:, 1]  # 50% chance to do y-horizontal flip
                if np.random.random() < 0.2: # 20% chance do random rotate along z axis
                    coords = self.rotate_point_cloud_z(np.expand_dims(coords, axis=0))
                    coords = np.squeeze(coords, axis=0)
                if np.random.random() < 0.03: # 3% chance to do random rotate along y axis
                    angle = np.random.randint(-90, 90)
                    coords = self.rotate_points_along_y(coords, np.deg2rad(angle))       
                if np.random.random() < 0.05: # 5% chance to do random rotate
                    angle = np.random.randint(-45, 45)
                    coords = self.rotate_point_cloud_by_angle(np.expand_dims(coords, axis=0), np.deg2rad(angle))
                    coords = np.squeeze(coords, axis=0)

        if self.jitter:
            coords = self.jitter_point_cloud(coords)
        
        if self.normalize:
            coords_norm = self.normalize_point_cloud(coords)            

        choice = np.random.choice(label.shape[0], self.num_points, replace=True)
        coords = coords[choice, :].transpose()  # [num_pc, xyz] to [xyz, num_pc]
        coords_norm = coords_norm[choice, :].transpose()
    
        if self.with_normal:
            normal = normal[choice, :].transpose()  # [num_pc, rgb] to [rgb, num_pc]
            if self.normalize:
                point_set = np.concatenate([coords, normal, coords_norm])  # point_set [xyz, rgb, normalize_xyz]
            else:
                point_set = np.concatenate([coords, normal])   # point_set [xyz, rgb]
        else:
            if self.normalize:
                point_set = np.concatenate([coords, coords_norm])
            else:
                point_set = coords
        
        return point_set, label[choice].transpose()  

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def normalize_point_cloud(points):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        return points / np.max(np.linalg.norm(points, axis=1))

    @staticmethod
    def jitter_point_cloud(points, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              3xN array, original batch of point clouds
            Return:
              3xN array, jittered batch of point clouds
        """
        assert (clip > 0)
        return np.clip(sigma * np.random.randn(*points.shape), -1 * clip, clip).astype(np.float32) + points
    
    #data augmentation
    def rotate_points_along_y(self, features, rotation_angle):
        """
        :param features: numpy array (N,C), first 3 channels are XYZ-coords
                         z is facing forward, x is left ward, y is downward
        :param rotation_angle: float, from z to x axis, unit: rad (rotate axis from z to x = rotate coords from x to z)
        :return:
            features: numpy array (N, C) with [0, 2] rotated
        """
        v_cos = np.cos(rotation_angle)
        v_sin = np.sin(rotation_angle)
        rotation_matrix_transpose = [[v_cos, v_sin], [-v_sin, v_cos]]
        features[:, [0, 2]] = np.dot(features[:, [0, 2]], rotation_matrix_transpose)
        return features
    
    def rotate_point_cloud_z(self, batch_data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, sinval, 0],
                                        [-sinval, cosval, 0],
                                        [0, 0, 1]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data
    
    def rotate_point_cloud_by_angle(self, batch_data, rotation_angle):
        """ Rotate the point cloud along up direction with certain angle.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k,:,0:3]
            rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data
    

class Custom(dict):
    def __init__(self, root, num_points, split=None, with_normal=True, data_aug=True,
                 normalize=True, jitter=True):
        super().__init__()
        if split is None:
            split = ['train', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = Custom_Dataset(root=root, num_points=num_points, split=s,
                                       with_normal=with_normal, data_aug=data_aug,
                                       normalize=normalize, jitter=jitter if s == 'train' else False)

