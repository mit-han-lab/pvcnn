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
        configs.train.deep_mutual_learning = configs.train.get('deep_mutual_learning', True)
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
        configs.train.best_checkpoint_path = os.path.join(save_path, 'best.pth.tar')
        best_checkpoints_dir = os.path.join(save_path, 'best')
        configs.train.best_checkpoint_paths = {
            m: os.path.join(best_checkpoints_dir, 'best.{}.pth.tar'.format(m.replace('/', '.')))
            for m in configs.train.metrics
        }
        os.makedirs(os.path.dirname(configs.train.checkpoints_path), exist_ok=True)
        os.makedirs(best_checkpoints_dir, exist_ok=True)
        if configs.train.deep_mutual_learning:
            configs.train.best_student_checkpoint_path = os.path.join(save_path, 'best_student.pth.tar')
            best_student_checkpoints_dir = os.path.join(save_path, 'best_student')
            configs.train.best_student_checkpoint_paths = {
                m: os.path.join(best_student_checkpoints_dir, 'best.{}.pth.tar'.format(m.replace('/', '.')))
                for m in configs.train.metrics
            }
            os.makedirs(best_student_checkpoints_dir, exist_ok=True)
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

    from modules import KLLoss

    ################################
    # Train / Eval Kernel Function #
    ################################

    # train kernel
    def train(model, loader, criterion, optimizer, scheduler, current_step, writer, schedule_per_epoch=True,
              model_student=None, criterion_dml=None, optimizer_student=None):
        model.train()
        if model_student is not None:
            model_student.train()
        for inputs, targets in tqdm(loader, desc='train', ncols=0):
            inputs = inputs.to(configs.device, non_blocking=True)
            targets = targets.to(configs.device, non_blocking=True)
            if model_student is None:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                writer.add_scalar('loss/train', loss.item(), current_step)
                current_step += targets.size(0)
                loss.backward()
                optimizer.step()
            else:
                # deep mutual learning
                optimizer.zero_grad()
                optimizer_student.zero_grad()
                outputs = model(inputs)
                outputs_student = model_student(inputs)
                loss = criterion(outputs, targets) + criterion_dml(outputs_student, outputs)
                loss_student = criterion(outputs_student, targets) + criterion_dml(outputs, outputs_student)
                writer.add_scalar('loss/train', loss.item(), current_step)
                writer.add_scalar('loss/train_student', loss_student.item(), current_step)
                current_step += targets.size(0)
                loss.backward()
                optimizer.step()
                loss_student.backward()
                optimizer_student.step()
            if not schedule_per_epoch and scheduler is not None:
                scheduler.step()
        if schedule_per_epoch and scheduler is not None:
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
    if configs.train.deep_mutual_learning:
        model_student = configs.model()
        if configs.device == 'cuda':
            model_student = torch.nn.DataParallel(model_student)
        model_student = model_student.to(configs.device)
        optimizer_student = configs.train.optimizer(model_student.parameters())
        criterion_dml = KLLoss()
    else:
        model_student, optimizer_student, criterion_dml = None, None, None

    last_epoch = -1
    best_metrics = {m: None for m in configs.train.metrics}
    best_metrics_student = {m: None for m in configs.train.metrics}
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

        if configs.train.deep_mutual_learning:
            if 'model_student' in checkpoint and checkpoint['model_student'] is not None:
                print(' => loading model_student')
                model_student.load_state_dict(checkpoint.pop('model_student'))
            if 'optimizer_student' in checkpoint and checkpoint['optimizer_student'] is not None:
                print(' => loading optimizer_student')
                optimizer_student.load_state_dict(checkpoint.pop('optimizer_student'))
            meters_student = checkpoint.get('meters_student', {})
            for m in configs.train.metrics:
                best_metrics_student[m] = meters_student.get(m + '_best', best_metrics_student[m])

    if 'scheduler' in configs.train and configs.train.scheduler is not None:
        scheduler_unit = configs.train.get('scheduler_unit', 'epoch')
        schedule_per_epoch = (scheduler_unit is None) or (scheduler_unit == 'epoch')
        configs.train.scheduler_unit = 'epoch' if schedule_per_epoch else 'iter'
        last_unit = last_epoch
        if not schedule_per_epoch:
            last_unit = (last_epoch + 1) * len(loaders['train']) - 1
        if 'T_max' in configs.train.scheduler:
            configs.train.scheduler.T_max = configs.train.num_epochs
            if not schedule_per_epoch:
                configs.train.scheduler.T_max *= len(loaders['train'])
            print(f'==> modify scheduler T_max to {configs.train.scheduler.T_max} {configs.train.scheduler_unit}s')
        print(f'==> creating scheduler "{configs.train.scheduler}" from {last_unit} {configs.train.scheduler_unit}')
        scheduler = configs.train.scheduler(optimizer, last_epoch=last_unit)
    else:
        scheduler = None
        schedule_per_epoch = True

    ############
    # Training #
    ############

    if last_epoch >= configs.train.num_epochs:
        meters, meters_student = dict(), dict()
        for split, loader in loaders.items():
            if split != 'train':
                meters.update(evaluate(model, loader=loader, split=split))
                if configs.train.deep_mutual_learning:
                    meters_student.update(evaluate(model_student, loader=loader, split=split))
        for k, meter in meters.items():
            print(f'[{k}] = {meter:2f}')
        for k, meter in meters_student.items():
            print(f'[{k}_student] = {meter:2f}')
        return

    with tensorboardX.SummaryWriter(configs.train.save_path) as writer:
        for current_epoch in range(last_epoch + 1, configs.train.num_epochs):
            current_step = current_epoch * len(dataset['train'])

            # train
            print(f'\n==> training epoch {current_epoch}/{configs.train.num_epochs}')
            train(model, loader=loaders['train'], criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                  current_step=current_step, writer=writer, schedule_per_epoch=schedule_per_epoch,
                  model_student=model_student, criterion_dml=criterion_dml, optimizer_student=optimizer_student)
            current_step += len(dataset['train'])

            # evaluate
            meters, meters_student = dict(), dict()
            for split, loader in loaders.items():
                if split != 'train':
                    meters.update(evaluate(model, loader=loader, split=split))
                    if configs.train.deep_mutual_learning:
                        meters_student.update(evaluate(model_student, loader=loader, split=split))

            # check whether it is the best
            best, best_student = {m: False for m in configs.train.metrics}, {m: False for m in configs.train.metrics}
            for m in configs.train.metrics:
                if best_metrics[m] is None or best_metrics[m] < meters[m]:
                    best_metrics[m], best[m] = meters[m], True
                meters[m + '_best'] = best_metrics[m]
                if configs.train.deep_mutual_learning:
                    if best_metrics_student[m] is None or best_metrics_student[m] < meters_student[m]:
                        best_metrics_student[m], best_student[m] = meters_student[m], True
                    meters_student[m + '_best'] = best_metrics_student[m]
            # log in tensorboard
            for k, meter in meters.items():
                print(f'[{k}] = {meter:2f}')
                writer.add_scalar(k, meter, current_step)
            for k, meter in meters_student.items():
                print(f'[{k}_student] = {meter:2f}')
                writer.add_scalar(k + '_student', meter, current_step)

            # save checkpoint
            torch.save({
                'epoch': current_epoch,
                'model': model.state_dict(),
                'model_student': model_student.state_dict() if configs.train.deep_mutual_learning else None,
                'optimizer': optimizer.state_dict(),
                'optimizer_student': optimizer.state_dict() if configs.train.deep_mutual_learning else None,
                'meters': meters,
                'meters_student': meters_student,
                'configs': configs,
            }, configs.train.checkpoint_path)
            shutil.copyfile(configs.train.checkpoint_path, configs.train.checkpoints_path.format(current_epoch))
            for m in configs.train.metrics:
                if best[m]:
                    shutil.copyfile(configs.train.checkpoint_path, configs.train.best_checkpoint_paths[m])
                if best_student[m]:
                    shutil.copyfile(configs.train.checkpoint_path, configs.train.best_student_checkpoint_paths[m])
            if best.get(configs.train.metric, False):
                shutil.copyfile(configs.train.checkpoint_path, configs.train.best_checkpoint_path)
            if best_student.get(configs.train.metric, False):
                shutil.copyfile(configs.train.checkpoint_path, configs.train.best_student_checkpoint_path)
            print(f'[save_path] = {configs.train.save_path}')


if __name__ == '__main__':
    main()
