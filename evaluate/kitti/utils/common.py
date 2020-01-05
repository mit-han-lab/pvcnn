# ref: https://github.com/traveller59/kitti-object-eval-python/blob/master/kitti_common.py

import pathlib
import re

import numpy as np

from .eval import get_official_eval_result

__all__ = ['eval_from_files']


def get_label_annotation(label_path):
    annotations = dict()
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    annotations['name'] = np.array([x[0] for x in content])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros([len(annotations['bbox'])])
    return annotations


def get_label_annotations(label_folder, image_ids=None):
    if image_ids is None:
        file_paths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        file_paths = filter(lambda f: prog.match(f.name), file_paths)
        image_ids = [int(p.stem) for p in file_paths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annotations = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        image_idx = f'{idx:06d}'
        label_filename = label_folder / (image_idx + '.txt')
        annotations.append(get_label_annotation(label_filename))
    return annotations


def eval_from_files(prediction_folder, ground_truth_folder, image_ids=None, verbose=False):
    prediction_annotations = get_label_annotations(prediction_folder)
    if isinstance(image_ids, str):
        with open(image_ids, 'r') as f:
            lines = f.readlines()
        image_ids = [int(line) for line in lines]
    ground_truth_annotations = get_label_annotations(ground_truth_folder, image_ids=image_ids)
    metrics, results, results_str = get_official_eval_result(
        gt_annos=ground_truth_annotations, dt_annos=prediction_annotations, current_classes=[0, 1, 2]
    )
    if verbose:
        print(results_str)
    return metrics, results
