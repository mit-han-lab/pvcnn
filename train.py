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

    print('==> loading configs from {}'.format(args.configs))
    configs.update_from_modules(*args.configs)
    if args.evaluate and configs.evaluate is not None:
        configs.train.batch_size = 10
        configs.dataset.split = 'test'
    else:
        configs.evaluate = None

    # define save path
    save_path = get_save_path(*args.configs, prefix='runs')
    os.makedirs(save_path, exist_ok=True)
    configs.train.save_path = save_path
    configs.train.checkpoint_path = os.path.join(save_path, 'latest.pth.tar')
    configs.train.best_checkpoint_path = os.path.join(save_path, 'best.pth.tar')

    # override configs with args
    configs.update_from_arguments(*opts)
    if len(gpus) == 0:
        configs.device = 'cpu'
        configs.device_ids = []
    else:
        configs.device = 'cuda'
        configs.device_ids = gpus
    configs.train.stats_path = configs.train.best_checkpoint_path.replace('pth.tar', 'eval.npy')

    return configs


def main():
    configs = prepare()
    if configs.evaluate is not None:
        configs.evaluate(configs)
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
            inputs = inputs.to(configs.device, non_blocking=True)
            targets = targets.to(configs.device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            writer.add_scalar('loss/train', loss.item(), current_step)
            current_step += targets.size(0)
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
                inputs = inputs.to(configs.device, non_blocking=True)
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
    if 'seed' in configs and configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        if configs.device == 'cuda' and configs.get('deterministic', True):
            cudnn.deterministic = True
            cudnn.benchmark = False

    print(configs)

    #####################################################################
    # Initialize DataLoaders, Model, Criterion, LRScheduler & Optimizer #
    #####################################################################

    print('\n==> loading dataset "{}"'.format(configs.dataset))
    dataset = configs.dataset()
    loaders = {}
    for split in dataset:
        loaders[split] = DataLoader(
            dataset[split], shuffle=(split == 'train'), batch_size=configs.train.batch_size,
            num_workers=configs.data.num_workers, pin_memory=True,
            # fixme: a quick fix for numpy random seed
            worker_init_fn=lambda x: np.random.seed()
        )

    print('\n==> creating model "{}"'.format(configs.model))
    model = configs.model()
    if configs.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(configs.device)
    criterion = configs.train.criterion().to(configs.device)
    optimizer = configs.train.optimizer(model.parameters())

    last_epoch, best_metric = -1, None
    if os.path.exists(configs.train.checkpoint_path):
        print('==> loading checkpoint "{}"'.format(configs.train.checkpoint_path))
        checkpoint = torch.load(configs.train.checkpoint_path)
        print(' => loading model')
        model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            print(' => loading optimizer')
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        last_epoch = checkpoint.get('epoch', last_epoch)
        best_metric = checkpoint.get('meters', {}).get('{}_best'.format(configs.train.metric), best_metric)

    if 'scheduler' in configs.train and configs.train.scheduler is not None:
        configs.train.scheduler.last_epoch = last_epoch
        print('==> creating scheduler "{}"'.format(configs.train.scheduler))
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
            print('[{}] = {:2f}'.format(k, meter))
        return

    with tensorboardX.SummaryWriter(configs.train.save_path) as writer:
        for current_epoch in range(last_epoch + 1, configs.train.num_epochs):
            current_step = current_epoch * len(dataset['train'])

            # train
            print('\n==> training epoch {}/{}'.format(current_epoch, configs.train.num_epochs))
            train(model, loader=loaders['train'], criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                  current_step=current_step, writer=writer)
            current_step += len(dataset['train'])

            # evaluate
            meters = dict()
            for split, loader in loaders.items():
                if split != 'train':
                    meters.update(evaluate(model, loader=loader, split=split))

            # check whether it is the best
            best = False
            if 'metric' in configs.train and configs.train.metric is not None:
                if best_metric is None or best_metric < meters[configs.train.metric]:
                    best_metric, best = meters[configs.train.metric], True
                meters[configs.train.metric + '_best'] = best_metric
            # log in tensorboard
            for k, meter in meters.items():
                print('[{}] = {:2f}'.format(k, meter))
                writer.add_scalar(k, meter, current_step)

            # save checkpoint
            torch.save({
                'epoch': current_epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'meters': meters,
                'configs': configs,
            }, configs.train.checkpoint_path)
            if best:
                shutil.copyfile(configs.train.checkpoint_path, configs.train.best_checkpoint_path)
            print('[save_path] = {}'.format(configs.train.save_path))


if __name__ == '__main__':
    main()
