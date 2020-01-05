import os

import h5py
import numpy as np
from torch.utils.data import Dataset

__all__ = ['S3DIS']


class _S3DISDataset(Dataset):
    def __init__(self, root, num_points, split='train', with_normalized_coords=True, holdout_area=5):
        """
        :param root: directory path to the s3dis dataset
        :param num_points: number of points to process for each scene
        :param split: 'train' or 'test'
        :param with_normalized_coords: whether include the normalized coords in features (default: True)
        :param holdout_area: which area to holdout (default: 5)
        """
        assert split in ['train', 'test']
        self.root = root
        self.split = split
        self.num_points = num_points
        self.holdout_area = None if holdout_area is None else int(holdout_area)
        self.with_normalized_coords = with_normalized_coords
        # keep at most 20/30 files in memory
        self.cache_size = 20 if split == 'train' else 30
        self.cache = {}

        # mapping batch index to corresponding file
        areas = []
        if self.split == 'train':
            for a in range(1, 7):
                if a != self.holdout_area:
                    areas.append(os.path.join(self.root, f'Area_{a}'))
        else:
            areas.append(os.path.join(self.root, f'Area_{self.holdout_area}'))

        self.num_scene_windows, self.max_num_points = 0, 0
        index_to_filename, scene_list = [], {}
        filename_to_start_index = {}
        for area in areas:
            area_scenes = os.listdir(area)
            area_scenes.sort()
            for scene in area_scenes:
                current_scene = os.path.join(area, scene)
                scene_list[current_scene] = []
                for split in ['zero', 'half']:
                    current_file = os.path.join(current_scene, f'{split}_0.h5')
                    filename_to_start_index[current_file] = self.num_scene_windows
                    h5f = h5py.File(current_file, 'r')
                    num_windows = h5f['data'].shape[0]
                    self.num_scene_windows += num_windows
                    for i in range(num_windows):
                        index_to_filename.append(current_file)
                    scene_list[current_scene].append(current_file)
        self.index_to_filename = index_to_filename
        self.filename_to_start_index = filename_to_start_index
        self.scene_list = scene_list

    def __len__(self):
        return self.num_scene_windows

    def __getitem__(self, index):
        filename = self.index_to_filename[index]
        if filename not in self.cache.keys():
            h5f = h5py.File(filename, 'r')
            scene_data = h5f['data']
            scene_label = h5f['label_seg']
            scene_num_points = h5f['data_num']
            if len(self.cache.keys()) < self.cache_size:
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
            else:
                victim_idx = np.random.randint(0, self.cache_size)
                cache_keys = list(self.cache.keys())
                cache_keys.sort()
                self.cache.pop(cache_keys[victim_idx])
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
        else:
            scene_data, scene_label, scene_num_points = self.cache[filename]

        internal_pos = index - self.filename_to_start_index[filename]
        current_window_data = np.array(scene_data[internal_pos]).astype(np.float32)
        current_window_label = np.array(scene_label[internal_pos]).astype(np.int64)
        current_window_num_points = scene_num_points[internal_pos]

        choices = np.random.choice(current_window_num_points, self.num_points,
                                   replace=(current_window_num_points < self.num_points))
        data = current_window_data[choices, ...].transpose()
        label = current_window_label[choices]
        # data[9, num_points] = [x_in_block, y_in_block, z_in_block, r, g, b, x / x_room, y / y_room, z / z_room]
        if self.with_normalized_coords:
            return data, label
        else:
            return data[:-3, :], label


class S3DIS(dict):
    def __init__(self, root, num_points, split=None, with_normalized_coords=True, holdout_area=5):
        super().__init__()
        if split is None:
            split = ['train', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = _S3DISDataset(root=root, num_points=num_points, split=s,
                                    with_normalized_coords=with_normalized_coords, holdout_area=holdout_area)
