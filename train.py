import argparse
import os
import random
import shutil


def prepare():
    from utils.common import get_save_path
    from utils.config import configs
    from utils.device import set_cuda_visible_devices

    # since PyTorch jams device selection, we have to parse args before import torch (issue #26790)
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--devices', default=None)
    parser.add_argument('--evaluate', default=False, action='store_true')
    args, opts = parser.parse_known_args()
    if args.devices is not None and args.devices != 'cpu':
        gpus = set_cuda_visible_devices(args.devices)
    else:
        gpus = []

    print(f'==> loading configs from {args.configs}')
    configs.update_from_modules(*args.configs)
    # define save path
    configs.train.save_path = get_save_path(*args.configs, prefix='runs')

    # override configs with args
    configs.update_from_arguments(*opts)
    if len(gpus) == 0:
        configs.device = 'cpu'
        configs.device_ids = []
    else:
        configs.device = 'cuda'
        configs.device_ids = gpus
    if args.evaluate and configs.evaluate.fn is not None:
        if 'dataset' in configs.evaluate:
            for k, v in configs.evaluate.dataset.items():
                configs.dataset[k] = v
    else:
        configs.evaluate = None

    if configs.evaluate is None:
        metrics = []
        if 'metric' in configs.train and configs.train.metric is not None:
            metrics.append(configs.train.metric)
        if 'metrics' in configs.train and configs.train.metrics is not None:
            for m in configs.train.metrics:
                if m not in metrics:
                    metrics.append(m)
        configs.train.metrics = metrics
        configs.train.metric = None if len(metrics) == 0 else metrics[0]

        save_path = configs.train.save_path
        configs.train.checkpoint_path = os.path.join(save_path, 'latest.pth.tar')
        configs.train.checkpoints_path = os.path.join(save_path, 'latest', 'e{}.pth.tar')
        configs.train.best_checkpoint_path = os.path.join(configs.train.save_path, 'best.pth.tar')
        best_checkpoints_dir = os.path.join(save_path, 'best')
        configs.train.best_checkpoint_paths = {
            m: os.path.join(best_checkpoints_dir, 'best.{}.pth.tar'.format(m.replace('/', '.')))
            for m in configs.train.metrics
        }
        os.makedirs(os.path.dirname(configs.train.checkpoints_path), exist_ok=True)
        os.makedirs(best_checkpoints_dir, exist_ok=True)
    else:
        if 'best_checkpoint_path' not in configs.evaluate or configs.evaluate.best_checkpoint_path is None:
            if 'best_checkpoint_path' in configs.train and configs.train.best_checkpoint_path is not None:
                configs.evaluate.best_checkpoint_path = configs.train.best_checkpoint_path
            else:
                configs.evaluate.best_checkpoint_path = os.path.join(configs.train.save_path, 'best.pth.tar')
        assert configs.evaluate.best_checkpoint_path.endswith('.pth.tar')
        configs.evaluate.predictions_path = configs.evaluate.best_checkpoint_path.replace('.pth.tar', '.predictions')
        configs.evaluate.stats_path = configs.evaluate.best_checkpoint_path.replace('.pth.tar', '.eval.npy')

    return configs


def main():
    configs = prepare()
    if configs.evaluate is not None:
        configs.evaluate.fn(configs)
        return

    import numpy as np
    import tensorboardX
    import torch
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    ################################
    # Train / Eval Kernel Function #
    ################################

    # train kernel
    def train(model, loader, criterion, optimizer, scheduler, current_step, writer):
        model.train()
        for inputs, targets in tqdm(loader, desc='train', ncols=0):
            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    batch_size = v.size(0)
                    inputs[k] = v.to(configs.device, non_blocking=True)
            else:
                batch_size = inputs.size(0)
                inputs = inputs.to(configs.device, non_blocking=True)
            if isinstance(targets, dict):
                for k, v in targets.items():
                    targets[k] = v.to(configs.device, non_blocking=True)
            else:
                targets = targets.to(configs.device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            writer.add_scalar('loss/train', loss.item(), current_step)
            current_step += batch_size
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # evaluate kernel
    def evaluate(model, loader, split='test'):
        meters = {}
        for k, meter in configs.train.meters.items():
            meters[k.format(split)] = meter()
        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=split, ncols=0):
                if isinstance(inputs, dict):
                    for k, v in inputs.items():
                        inputs[k] = v.to(configs.device, non_blocking=True)
                else:
                    inputs = inputs.to(configs.device, non_blocking=True)
                if isinstance(targets, dict):
                    for k, v in targets.items():
                        targets[k] = v.to(configs.device, non_blocking=True)
                else:
                    targets = targets.to(configs.device, non_blocking=True)
                outputs = model(inputs)
                for meter in meters.values():
                    meter.update(outputs, targets)
        for k, meter in meters.items():
            meters[k] = meter.compute()
        return meters

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

    #####################################################################
    # Initialize DataLoaders, Model, Criterion, LRScheduler & Optimizer #
    #####################################################################

    print(f'\n==> loading dataset "{configs.dataset}"')
    dataset = configs.dataset()
    loaders = {}
    for split in dataset:
        loaders[split] = DataLoader(
            dataset[split], shuffle=(split == 'train'), batch_size=configs.train.batch_size,
            num_workers=configs.data.num_workers, pin_memory=True,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
        )

    print(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    if configs.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(configs.device)
    criterion = configs.train.criterion().to(configs.device)
    optimizer = configs.train.optimizer(model.parameters())

    last_epoch, best_metrics = -1, {m: None for m in configs.train.metrics}
    if os.path.exists(configs.train.checkpoint_path):
        print(f'==> loading checkpoint "{configs.train.checkpoint_path}"')
        checkpoint = torch.load(configs.train.checkpoint_path)
        print(' => loading model')
        model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            print(' => loading optimizer')
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        last_epoch = checkpoint.get('epoch', last_epoch)
        meters = checkpoint.get('meters', {})
        for m in configs.train.metrics:
            best_metrics[m] = meters.get(m + '_best', best_metrics[m])
        del checkpoint

    if 'scheduler' in configs.train and configs.train.scheduler is not None:
        configs.train.scheduler.last_epoch = last_epoch
        print(f'==> creating scheduler "{configs.train.scheduler}"')
        scheduler = configs.train.scheduler(optimizer)
    else:
        scheduler = None

    ############
    # Training #
    ############

    if last_epoch >= configs.train.num_epochs:
        meters = dict()
        for split, loader in loaders.items():
            if split != 'train':
                meters.update(evaluate(model, loader=loader, split=split))
        for k, meter in meters.items():
            print(f'[{k}] = {meter:2f}')
        return

    with tensorboardX.SummaryWriter(configs.train.save_path) as writer:
        for current_epoch in range(last_epoch + 1, configs.train.num_epochs):
            current_step = current_epoch * len(dataset['train'])

            # train
            print(f'\n==> training epoch {current_epoch}/{configs.train.num_epochs}')
            train(model, loader=loaders['train'], criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                  current_step=current_step, writer=writer)
            current_step += len(dataset['train'])

            # evaluate
            meters = dict()
            for split, loader in loaders.items():
                if split != 'train':
                    meters.update(evaluate(model, loader=loader, split=split))

            # check whether it is the best
            best = {m: False for m in configs.train.metrics}
            for m in configs.train.metrics:
                if best_metrics[m] is None or best_metrics[m] < meters[m]:
                    best_metrics[m], best[m] = meters[m], True
                meters[m + '_best'] = best_metrics[m]
            # log in tensorboard
            for k, meter in meters.items():
                print(f'[{k}] = {meter:2f}')
                writer.add_scalar(k, meter, current_step)

            # save checkpoint
            torch.save({
                'epoch': current_epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'meters': meters,
                'configs': configs,
            }, configs.train.checkpoint_path)
            shutil.copyfile(configs.train.checkpoint_path, configs.train.checkpoints_path.format(current_epoch))
            for m in configs.train.metrics:
                if best[m]:
                    shutil.copyfile(configs.train.checkpoint_path, configs.train.best_checkpoint_paths[m])
            if best.get(configs.train.metric, False):
                shutil.copyfile(configs.train.checkpoint_path, configs.train.best_checkpoint_path)
            print(f'[save_path] = {configs.train.save_path}')


if __name__ == '__main__':
    main()
