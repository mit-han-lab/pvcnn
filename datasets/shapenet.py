import json
import os

import numpy as np
from torch.utils.data import Dataset

__all__ = ['ShapeNet']


class _ShapeNetDataset(Dataset):
    def __init__(self, root, num_points, split='train', with_normal=True, with_one_hot_shape_id=True,
                 normalize=True, jitter=True):
        assert split in ['train', 'test']
        self.root = root
        self.num_points = num_points
        self.split = split
        self.with_normal = with_normal
        self.with_one_hot_shape_id = with_one_hot_shape_id
        self.normalize = normalize
        self.jitter = jitter

        shape_dir_to_shape_id = {}
        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for shape_id, line in enumerate(f):
                shape_name, shape_dir = line.strip().split()
                shape_dir_to_shape_id[shape_dir] = shape_id
        file_paths = []
        if self.split == 'train':
            split = ['train', 'val']
        else:
            split = ['test']
        for s in split:
            with open(os.path.join(self.root, 'train_test_split', f'shuffled_{s}_file_list.json'), 'r') as f:
                file_list = json.load(f)
                for file_path in file_list:
                    _, shape_dir, filename = file_path.split('/')
                    file_paths.append(
                        (os.path.join(self.root, shape_dir, filename + '.txt'),
                         shape_dir_to_shape_id[shape_dir])
                    )
        self.file_paths = file_paths
        self.num_shapes = 16
        self.num_classes = 50

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            coords, normal, label, shape_id = self.cache[index]
        else:
            file_path, shape_id = self.file_paths[index]
            data = np.loadtxt(file_path).astype(np.float32)
            coords = data[:, :3]
            if self.normalize:
                coords = self.normalize_point_cloud(coords)
            normal = data[:, 3:6]
            label = data[:, -1].astype(np.int64)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (coords, normal, label, shape_id)

        choice = np.random.choice(label.shape[0], self.num_points, replace=True)
        coords = coords[choice, :].transpose()
        if self.jitter:
            coords = self.jitter_point_cloud(coords)
        if self.with_normal:
            normal = normal[choice, :].transpose()
            if self.with_one_hot_shape_id:
                shape_one_hot = np.zeros((self.num_shapes, self.num_points), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, normal, shape_one_hot])
            else:
                point_set = np.concatenate([coords, normal])
        else:
            if self.with_one_hot_shape_id:
                shape_one_hot = np.zeros((self.num_shapes, self.num_points), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, shape_one_hot])
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


class ShapeNet(dict):
    def __init__(self, root, num_points, split=None, with_normal=True, with_one_hot_shape_id=True,
                 normalize=True, jitter=True):
        super().__init__()
        if split is None:
            split = ['train', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = _ShapeNetDataset(root=root, num_points=num_points, split=s,
                                       with_normal=with_normal, with_one_hot_shape_id=with_one_hot_shape_id,
                                       normalize=normalize, jitter=jitter if s == 'train' else False)
