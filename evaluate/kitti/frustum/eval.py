import argparse
import os
import random
import shutil
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
    configs.train.best_checkpoint_path = os.path.join(save_path, 'best.pth.tar')

    # override configs with args
    configs.update_from_arguments(*opts)
    if len(gpus) == 0:
        configs.device = 'cpu'
        configs.device_ids = []
    else:
        configs.device = 'cuda'
        configs.device_ids = gpus
    if 'dataset' in configs.evaluate:
        for k, v in configs.evaluate.dataset.items():
            configs.dataset[k] = v
    if 'best_checkpoint_path' not in configs.evaluate or configs.evaluate.best_checkpoint_path is None:
        if 'best_checkpoint_path' in configs.train and configs.train.best_checkpoint_path is not None:
            configs.evaluate.best_checkpoint_path = configs.train.best_checkpoint_path
        else:
            configs.evaluate.best_checkpoint_path = os.path.join(configs.train.save_path, 'best.pth.tar')
    assert configs.evaluate.best_checkpoint_path.endswith('.pth.tar')
    configs.evaluate.predictions_path = configs.evaluate.best_checkpoint_path.replace('.pth.tar', '.predictions')
    configs.evaluate.stats_path = configs.evaluate.best_checkpoint_path.replace('.pth.tar', '.eval.npy')

    return configs


def evaluate(configs=None):
    configs = prepare() if configs is None else configs

    import time

    import torch
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from ..utils import eval_from_files

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

    if configs.evaluate.num_tests > 1:
        results = dict()
        stats_path = os.path.join(configs.evaluate.stats_path.replace('.npy', '.t'), 'best.eval.t{}.npy')
        predictions_path = os.path.join(configs.evaluate.predictions_path + '.t', 'best.predictions.t{}')
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

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

    for test_index in range(configs.evaluate.num_tests):
        if test_index == 0:
            print(configs)

        seed = configs.seed
        if test_index > 0:
            seed = random.randint(1, int(time.time())) % (2 ** 32 - 1)
            print(f'\n==> Test [{test_index:02d}/{configs.evaluate.num_tests:02d}] initial seed\n[seed] = {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if configs.evaluate.num_tests > 1:
            configs.evaluate.stats_path = stats_path.format(test_index)
            configs.evaluate.predictions_path = predictions_path.format(test_index)

        if os.path.exists(configs.evaluate.stats_path):
            print(f'==> hit {configs.evaluate.stats_path}')
            predictions = np.load(configs.evaluate.stats_path)
            image_ids = write_predictions(configs.evaluate.predictions_path, ids=dataset.data.ids,
                                          classes=dataset.data.class_names, boxes_2d=dataset.data.boxes_2d,
                                          predictions=predictions,
                                          image_id_file_path=configs.evaluate.image_id_file_path)
            _, current_results = eval_from_files(prediction_folder=configs.evaluate.predictions_path,
                                                 ground_truth_folder=configs.evaluate.ground_truth_path,
                                                 image_ids=image_ids, verbose=True)
            if configs.evaluate.num_tests == 1:
                return
            else:
                for class_name, v in current_results.items():
                    if class_name not in results:
                        results[class_name] = dict()
                    for kind, r in v.items():
                        if kind not in results[class_name]:
                            results[class_name][kind] = []
                        results[class_name][kind].append(r)
                continue

        loader = DataLoader(
            dataset, shuffle=False, batch_size=configs.evaluate.batch_size,
            num_workers=configs.data.num_workers, pin_memory=True,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
        )

        ##############
        # Evaluation #
        ##############

        predictions = np.zeros((len(dataset), 8))
        size_templates = configs.data.size_templates.to(configs.device)
        heading_angle_bin_centers = torch.arange(
            0, 2 * np.pi, 2 * np.pi / configs.data.num_heading_angle_bins).to(configs.device)
        current_step = 0

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc='eval', ncols=0):
                for k, v in inputs.items():
                    inputs[k] = v.to(configs.device, non_blocking=True)
                outputs = model(inputs)

                center = outputs['center']  # (B, 3)
                heading_scores = outputs['heading_scores']  # (B, NH)
                heading_residuals = outputs['heading_residuals']  # (B, NH)
                size_scores = outputs['size_scores']  # (B, NS)
                size_residuals = outputs['size_residuals']  # (B, NS, 3)

                batch_size = center.size(0)
                batch_id = torch.arange(batch_size, device=center.device)
                heading_bin_id = torch.argmax(heading_scores, dim=1)
                heading = heading_angle_bin_centers[heading_bin_id] + heading_residuals[batch_id, heading_bin_id]  # (B, )
                size_template_id = torch.argmax(size_scores, dim=1)
                size = size_templates[size_template_id] + size_residuals[batch_id, size_template_id]  # (B, 3)

                center = center.cpu().numpy()
                heading = heading.cpu().numpy()
                size = size.cpu().numpy()
                rotation_angle = targets['rotation_angle'].cpu().numpy()  # (B, )
                rgb_score = targets['rgb_score'].cpu().numpy()  # (B, )

                update_predictions(predictions=predictions, center=center, heading=heading, size=size,
                                   rotation_angle=rotation_angle, rgb_score=rgb_score,
                                   current_step=current_step, batch_size=batch_size)
                current_step += batch_size

        np.save(configs.evaluate.stats_path, predictions)
        image_ids = write_predictions(configs.evaluate.predictions_path, ids=dataset.data.ids,
                                      classes=dataset.data.class_names, boxes_2d=dataset.data.boxes_2d,
                                      predictions=predictions, image_id_file_path=configs.evaluate.image_id_file_path)
        _, current_results = eval_from_files(prediction_folder=configs.evaluate.predictions_path,
                                             ground_truth_folder=configs.evaluate.ground_truth_path,
                                             image_ids=image_ids, verbose=True)
        if configs.evaluate.num_tests == 1:
            return
        else:
            for class_name, v in current_results.items():
                if class_name not in results:
                    results[class_name] = dict()
                for kind, r in v.items():
                    if kind not in results[class_name]:
                        results[class_name][kind] = []
                    results[class_name][kind].append(r)
    for class_name, v in results.items():
        print(f'{class_name}  AP(Average Precision)')
        for kind, r in v.items():
            r = np.asarray(r)
            m = r.mean(axis=0)
            s = r.std(axis=0)
            u = r.max(axis=0)
            rs = ', '.join(f'{mv:.2f} +/- {sv:.2f} ({uv:.2f})' for mv, sv, uv in zip(m, s, u))
            print(f'{kind:<4} AP: {rs}')


@numba.jit()
def update_predictions(predictions, center, heading, size, rotation_angle, rgb_score, current_step, batch_size):
    for b in range(batch_size):
        l, w, h = size[b]
        x, y, z = center[b]  # (3)
        r = rotation_angle[b]
        t = heading[b]
        s = rgb_score[b]
        v_cos = np.cos(r)
        v_sin = np.sin(r)
        cx = v_cos * x + v_sin * z  # it should be v_cos * x - v_sin * z, but the rotation angle = -r
        cy = y + h / 2.0
        cz = v_cos * z - v_sin * x  # it should be v_sin * x + v_cos * z, but the rotation angle = -r
        r = r + t
        while r > np.pi:
            r = r - 2 * np.pi
        while r < -np.pi:
            r = r + 2 * np.pi
        predictions[current_step + b] = [h, w, l, cx, cy, cz, r, s]


def write_predictions(prediction_path, ids, classes, boxes_2d, predictions, image_id_file_path=None):
    import pathlib

    # map from idx to list of strings, each string is a line (with \n)
    results = {}
    for i in range(predictions.shape[0]):
        idx = ids[i]
        output_str = ('{} -1 -1 -10 '
                      '{:f} {:f} {:f} {:f} '
                      '{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(classes[i], *boxes_2d[i][:4], *predictions[i]))
        if idx not in results:
            results[idx] = []
        results[idx].append(output_str)

    # write txt files
    if os.path.exists(prediction_path):
        shutil.rmtree(prediction_path)
    os.mkdir(prediction_path)
    for k, v in results.items():
        file_path = os.path.join(prediction_path, f'{k:06d}.txt')
        with open(file_path, 'w') as f:
            f.writelines(v)

    if image_id_file_path is not None and os.path.exists(image_id_file_path):
        with open(image_id_file_path, 'r') as f:
            val_ids = f.readlines()
        for idx in val_ids:
            idx = idx.strip()
            file_path = os.path.join(prediction_path, f'{idx}.txt')
            if not os.path.exists(file_path):
                # print(f'warning: {file_path} doesn\'t exist as indicated in {image_id_file_path}')
                pathlib.Path(file_path).touch()
        return image_id_file_path
    else:
        image_ids = sorted([k for k in results.keys()])
        return image_ids


if __name__ == '__main__':
    evaluate()
