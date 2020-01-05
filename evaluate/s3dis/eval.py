import argparse
import os
import random
import sys

import numba
import numpy as np

sys.path.append(os.getcwd())

__all__ = ['evaluate']


def prepare():
    from utils.common import get_save_path
    from utils.config import configs
    from utils.device import set_cuda_visible_devices

    # since PyTorch jams device selection, we have to parse args before import torch (issue #26790)
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--devices', default=None)
    args, opts = parser.parse_known_args()
    if args.devices is not None and args.devices != 'cpu':
        gpus = set_cuda_visible_devices(args.devices)
    else:
        gpus = []

    print(f'==> loading configs from {args.configs}')
    configs.update_from_modules(*args.configs)
    # define save path
    save_path = get_save_path(*args.configs, prefix='runs')
    os.makedirs(save_path, exist_ok=True)
    configs.train.save_path = save_path

    # override configs with args
    configs.update_from_arguments(*opts)
    if len(gpus) == 0:
        configs.device = 'cpu'
        configs.device_ids = []
    else:
        configs.device = 'cuda'
        configs.device_ids = gpus
    configs.dataset.split = configs.evaluate.dataset.split
    if 'best_checkpoint_path' not in configs.evaluate or configs.evaluate.best_checkpoint_path is None:
        if 'best_checkpoint_path' in configs.train and configs.train.best_checkpoint_path is not None:
            configs.evaluate.best_checkpoint_path = configs.train.best_checkpoint_path
        else:
            configs.evaluate.best_checkpoint_path = os.path.join(configs.train.save_path, 'best.pth.tar')
    assert configs.evaluate.best_checkpoint_path.endswith('.pth.tar')
    configs.evaluate.stats_path = configs.evaluate.best_checkpoint_path.replace('.pth.tar', '.eval.npy')

    return configs


def evaluate(configs=None):
    configs = prepare() if configs is None else configs

    import h5py
    import math
    import torch
    import torch.backends.cudnn as cudnn
    import torch.nn.functional as F
    from tqdm import tqdm

    #####################
    # Kernel Definition #
    #####################

    def print_stats(stats):
        stats = stats.sum(axis=-1)
        iou = stats[2] / (stats[0] + stats[1] - stats[2])
        print('classes: {}'.format('  '.join(map('{:>8d}'.format, stats[0].astype(np.int64)))))
        print('positiv: {}'.format('  '.join(map('{:>8d}'.format, stats[1].astype(np.int64)))))
        print('truepos: {}'.format('  '.join(map('{:>8d}'.format, stats[2].astype(np.int64)))))
        print('clssiou: {}'.format('  '.join(map('{:>8.2f}'.format, iou * 100))))
        print('meanAcc: {:4.2f}'.format(stats[2].sum() / stats[1].sum() * 100))
        print('meanIoU: {:4.2f}'.format(iou.mean() * 100))

    ###########
    # Prepare #
    ###########

    if configs.device == 'cuda':
        cudnn.benchmark = True
        if configs.get('deterministic', False):
            cudnn.deterministic = True
            cudnn.benchmark = False
    if ('seed' not in configs) or (configs.seed is None):
        configs.seed = torch.initial_seed() % (2 ** 32 - 1)
    seed = configs.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(configs)

    if os.path.exists(configs.evaluate.stats_path):
        stats = np.load(configs.evaluate.stats_path)
        print_stats(stats)
        return

    #################################
    # Initialize DataLoaders, Model #
    #################################

    print(f'\n==> loading dataset "{configs.dataset}"')
    dataset = configs.dataset()[configs.dataset.split]

    print(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    if configs.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(configs.device)

    if os.path.exists(configs.evaluate.best_checkpoint_path):
        print(f'==> loading checkpoint "{configs.evaluate.best_checkpoint_path}"')
        checkpoint = torch.load(configs.evaluate.best_checkpoint_path)
        model.load_state_dict(checkpoint.pop('model'))
        del checkpoint
    else:
        return

    model.eval()

    ##############
    # Evaluation #
    ##############

    total_num_scenes = len(dataset.scene_list)
    stats = np.zeros((3, configs.data.num_classes, total_num_scenes))

    for scene_index, (scene, scene_files) in enumerate(tqdm(dataset.scene_list.items(), desc='eval', ncols=0)):
        ground_truth = np.load(os.path.join(scene, 'label.npy')).reshape(-1)
        total_num_points_in_scene = ground_truth.shape[0]
        confidences = np.zeros(total_num_points_in_scene, dtype=np.float32)
        predictions = np.full(total_num_points_in_scene, -1, dtype=np.int64)

        for filename in scene_files:
            h5f = h5py.File(filename, 'r')
            scene_data = h5f['data'][...].astype(np.float32)
            scene_num_points = h5f['data_num'][...].astype(np.int64)
            window_to_scene_mapping = h5f['indices_split_to_full'][...].astype(np.int64)

            num_windows, max_num_points_per_window, num_channels = scene_data.shape
            extra_batch_size = configs.evaluate.num_votes * math.ceil(max_num_points_per_window / dataset.num_points)
            total_num_voted_points = extra_batch_size * dataset.num_points

            for min_window_index in range(0, num_windows, configs.evaluate.batch_size):
                max_window_index = min(min_window_index + configs.evaluate.batch_size, num_windows)
                batch_size = max_window_index - min_window_index
                window_data = scene_data[np.arange(min_window_index, max_window_index)]
                window_data = window_data.reshape(batch_size, -1, num_channels)

                # repeat, shuffle and tile
                # TODO: speedup here
                batched_inputs = np.zeros((batch_size, total_num_voted_points, num_channels), dtype=np.float32)
                batched_shuffled_point_indices = np.zeros((batch_size, total_num_voted_points), dtype=np.int64)
                for relative_window_index in range(batch_size):
                    num_points_in_window = scene_num_points[relative_window_index + min_window_index]
                    num_repeats = math.ceil(total_num_voted_points / num_points_in_window)
                    shuffled_point_indices = np.tile(np.arange(num_points_in_window), num_repeats)
                    shuffled_point_indices = shuffled_point_indices[:total_num_voted_points]
                    np.random.shuffle(shuffled_point_indices)
                    batched_shuffled_point_indices[relative_window_index] = shuffled_point_indices
                    batched_inputs[relative_window_index] = window_data[relative_window_index][shuffled_point_indices]

                # model inference
                inputs = torch.from_numpy(
                    batched_inputs.reshape((batch_size * extra_batch_size, dataset.num_points, -1)).transpose(0, 2, 1)
                ).float().to(configs.device)
                with torch.no_grad():
                    batched_confidences, batched_predictions = F.softmax(model(inputs), dim=1).max(dim=1)
                    batched_confidences = batched_confidences.view(batch_size, total_num_voted_points).cpu().numpy()
                    batched_predictions = batched_predictions.view(batch_size, total_num_voted_points).cpu().numpy()

                update_scene_predictions(batched_confidences, batched_predictions, batched_shuffled_point_indices,
                                         confidences, predictions, window_to_scene_mapping,
                                         total_num_voted_points, batch_size, min_window_index)

        # update stats
        update_stats(stats, ground_truth, predictions, scene_index, total_num_points_in_scene)

    np.save(configs.evaluate.stats_path, stats)
    print_stats(stats)


@numba.jit()
def update_scene_predictions(batched_confidences, batched_predictions, batched_shuffled_point_indices,
                             scene_confidences, scene_predictions, window_to_scene_mapping, total_num_voted_points,
                             batch_size, min_window_index):
    for b in range(batch_size):
        window_index = min_window_index + b
        current_window_mapping = window_to_scene_mapping[window_index]
        current_shuffled_point_indices = batched_shuffled_point_indices[b]
        current_confidences = batched_confidences[b]
        current_predictions = batched_predictions[b]
        for p in range(total_num_voted_points):
            point_index = current_window_mapping[current_shuffled_point_indices[p]]
            current_confidence = current_confidences[p]
            if current_confidence > scene_confidences[point_index]:
                scene_confidences[point_index] = current_confidence
                scene_predictions[point_index] = current_predictions[p]


@numba.jit()
def update_stats(stats, ground_truth, predictions, scene_index, total_num_points_in_scene):
    for p in range(total_num_points_in_scene):
        gt = int(ground_truth[p])
        pd = int(predictions[p])
        stats[0, gt, scene_index] += 1
        stats[1, pd, scene_index] += 1
        if gt == pd:
            stats[2, gt, scene_index] += 1


if __name__ == '__main__':
    evaluate()
