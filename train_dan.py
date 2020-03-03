import argparse
import math
import os
import random
import shutil

from tqdm import trange

from modules import mmd
from modules.loss import discrepancy_loss
from utils.common import loop_iterable


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
        # if 'dataset' in configs.evaluate:
        #     for k, v in configs.evaluate.dataset.items():
        #         configs.dataset[k] = v
        pass
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
        configs.train.checkpoint_path = os.path.join(save_path,
                                                     'latest.pth.tar')
        configs.train.checkpoints_path = os.path.join(save_path, 'latest',
                                                      'e{}.pth.tar')
        configs.train.best_checkpoint_path = os.path.join(
            configs.train.save_path, 'best.pth.tar')
        best_checkpoints_dir = os.path.join(save_path, 'best')
        configs.train.best_checkpoint_paths = {
            m: os.path.join(best_checkpoints_dir,
                            'best.{}.pth.tar'.format(m.replace('/', '.')))
            for m in configs.train.metrics
        }
        os.makedirs(os.path.dirname(configs.train.checkpoints_path),
                    exist_ok=True)
        os.makedirs(best_checkpoints_dir, exist_ok=True)
    else:
        if 'best_checkpoint_path' not in configs.evaluate or configs.evaluate.best_checkpoint_path is None:
            if 'best_checkpoint_path' in configs.train and configs.train.best_checkpoint_path is not None:
                configs.evaluate.best_checkpoint_path = configs.train.best_checkpoint_path
            else:
                configs.evaluate.best_checkpoint_path = os.path.join(
                    configs.train.save_path, 'best.pth.tar')
        assert configs.evaluate.best_checkpoint_path.endswith('.pth.tar')
        configs.evaluate.predictions_path = configs.evaluate.best_checkpoint_path.replace(
            '.pth.tar', '.predictions')
        configs.evaluate.stats_path = configs.evaluate.best_checkpoint_path.replace(
            '.pth.tar', '.eval.npy')

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

    def adjust_learning_rate(optimizer, epoch, args_lr):
        """Sets the learning rate to the initial LR decayed by half by every 5 or 10 epochs"""
        if epoch > 0:
            if epoch <= 30:
                lr = args_lr * (0.5 ** (epoch // 5))
            else:
                lr = args_lr * (0.5 ** (epoch // 10))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            writer.add_scalar('lr_dis', lr, epoch)

    # train kernel
    def train(model, source_loader, target_loader, criterion, optimizer_g,
              optimizer_cls, optimizer_dis, scheduler_g, scheduler_cls,
              current_step, writer, cons):

        model.train()
        loss_total = 0
        loss_adv_total = 0
        loss_node_total = 0
        data_total = 0

        batch_iterator = zip(loop_iterable(source_loader),
                             loop_iterable(target_loader))

        for _ in trange(len(source_loader)):
            (inputs, targets), (inputs_t, _) = next(batch_iterator)

            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    batch_size = v.size(0)
                    inputs[k] = v.to(configs.device, non_blocking=True)
            else:
                batch_size = inputs.size(0)
                inputs = inputs.to(configs.device, non_blocking=True)

            if isinstance(inputs_t, dict):
                for k, v in inputs_t.items():
                    batch_size = v.size(0)
                    inputs_t[k] = v.to(configs.device, non_blocking=True)
            else:
                batch_size = inputs_t.size(0)
                inputs_t = inputs_t.to(configs.device, non_blocking=True)

            if isinstance(targets, dict):
                for k, v in targets.items():
                    targets[k] = v.to(configs.device, non_blocking=True)
            else:
                targets = targets.to(configs.device, non_blocking=True)

            outputs = model(inputs)

            pred_t1, pred_t2 = model.module.inst_seg_net(
                {'features': inputs_t['features'],
                 'one_hot_vectors': inputs_t['one_hot_vectors']},
                constant=cons, adaptation=True)

            loss_s = criterion(outputs, targets)

            # Adversarial loss
            loss_adv = - 1 * discrepancy_loss(pred_t1, pred_t2)

            loss = loss_s + loss_adv
            loss.backward()
            optimizer_g.step()
            optimizer_cls.step()
            optimizer_g.zero_grad()
            optimizer_cls.zero_grad()

            # Local Alignment
            feat_node_s = model.module.inst_seg_net({'features': inputs['features'],
                                              'one_hot_vectors': inputs[
                                                  'one_hot_vectors']},
                                             node_adaptation_s=True)

            feat_node_t = model.module.inst_seg_net({'features': inputs_t['features'],
                                              'one_hot_vectors': inputs_t[
                                                  'one_hot_vectors']},
                                             node_adaptation_t=True)

            sigma_list = [0.01, 0.1, 1, 10, 100]
            loss_node_adv = 1 * mmd.mix_rbf_mmd2(feat_node_s, feat_node_t,
                                                 sigma_list)
            loss = loss_node_adv

            loss.backward()
            optimizer_dis.step()
            optimizer_dis.zero_grad()

            loss_total += loss_s.item() * batch_size
            loss_adv_total += loss_adv.item() * batch_size
            loss_node_total += loss_node_adv.item() * batch_size
            data_total += batch_size

            writer.add_scalar('loss_s/train', loss_total / data_total, current_step)
            writer.add_scalar('loss_adv/train', loss_adv_total / data_total, current_step)
            writer.add_scalar('loss_node/train', loss_node_total / data_total, current_step)
            current_step += batch_size

        if scheduler_g is not None:
            scheduler_g.step()

        if scheduler_cls is not None:
            scheduler_cls.step()

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

    print(f'\n==> loading source dataset "{configs.source_dataset}"')
    source_dataset = configs.source_dataset()
    source_loaders = {"train": DataLoader(
        source_dataset["train"], shuffle=True,
        batch_size=configs.train.batch_size,
        num_workers=configs.data.num_workers, pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )}

    print(f'\n==> loading target dataset "{configs.target_dataset}"')
    target_dataset = configs.target_dataset()
    target_loaders = {}
    for split in target_dataset:
        target_loaders[split] = DataLoader(
            target_dataset[split], shuffle=(split == 'train'),
            batch_size=configs.train.batch_size,
            num_workers=configs.data.num_workers, pin_memory=True,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
        )

    print(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    if configs.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(configs.device)
    criterion = configs.train.criterion().to(configs.device)
    #params
    gen_params = [{'params':v} for k,v in model.module.inst_seg_net.g.named_parameters()
                  if 'pred_offset' not in k]

    cls_params = [{'params':model.module.inst_seg_net.c1.parameters()},
                  {'params':model.module.inst_seg_net.c2.parameters()},
                  {'params':model.module.center_reg_net.parameters()},
                  {'params':model.module.box_est_net.parameters()}]

    dis_params = [{'params':model.module.inst_seg_net.g.parameters()},
                  {'params':model.module.inst_seg_net.attention_s.parameters()},
                  {'params':model.module.inst_seg_net.attention_t.parameters()}]

    optimizer_g = configs.train.optimizer_g(gen_params)
    optimizer_cls = configs.train.optimizer_cls(cls_params)
    optimizer_dis = configs.train.optimizer_dis(dis_params)

    last_epoch, best_metrics = -1, {m: None for m in configs.train.metrics}

    if os.path.exists(configs.train.checkpoint_path):

        print(f'==> loading checkpoint "{configs.train.checkpoint_path}"')
        checkpoint = torch.load(configs.train.checkpoint_path)

        print(' => loading model')
        model.load_state_dict(checkpoint.pop('model'))

        if 'optimizer_g' in checkpoint and checkpoint['optimizer_g'] is not None:
            print(' => loading optimizer_g')
            optimizer_g.load_state_dict(checkpoint.pop('optimizer_g'))

        if 'optimizer_cls' in checkpoint and checkpoint['optimizer_cls'] is not None:
            print(' => loading optimizer_cls')
            optimizer_cls.load_state_dict(checkpoint.pop('optimizer_cls'))

        if 'optimizer_dis' in checkpoint and checkpoint['optimizer_dis'] is not None:
            print(' => loading optimizer_dis')
            optimizer_dis.load_state_dict(checkpoint.pop('optimizer_dis'))

        last_epoch = checkpoint.get('epoch', last_epoch)
        meters = checkpoint.get('meters', {})

        for m in configs.train.metrics:
            best_metrics[m] = meters.get(m + '_best', best_metrics[m])

        del checkpoint

    if 'scheduler_g' in configs.train and configs.train.scheduler_g is not None:
        configs.train.scheduler_g.last_epoch = last_epoch
        print(f'==> creating scheduler "{configs.train.scheduler_g}"')
        scheduler_g = configs.train.scheduler_g(optimizer_g)
    else:
        scheduler_g = None

    if 'scheduler_c' in configs.train and configs.train.scheduler_c is not None:
        configs.train.scheduler_c.last_epoch = last_epoch
        print(f'==> creating scheduler "{configs.train.scheduler_c}"')
        scheduler_c = configs.train.scheduler_c(optimizer_cls)
    else:
        scheduler_c = None

    ############
    # Training #
    ############

    if last_epoch >= configs.train.num_epochs:
        meters = dict()
        for split, loader in target_loaders.items():
            if split != 'train':
                meters.update(evaluate(model, loader=loader, split=split))
        for k, meter in meters.items():
            print(f'[{k}] = {meter:2f}')
        return

    with tensorboardX.SummaryWriter(configs.train.save_path) as writer:
        step_size = min(len(source_dataset['train']),
                        len(target_dataset['train']))

        for current_epoch in range(last_epoch + 1, configs.train.num_epochs):
            current_step = current_epoch * step_size
            cons = math.sin((current_epoch + 1)/configs.train.num_epochs * math.pi/2)
            adjust_learning_rate(optimizer_dis, current_epoch, configs.train.base_lr)

            writer.add_scalar('lr_g', scheduler_g.get_lr()[0], current_epoch)
            writer.add_scalar('lr_c', scheduler_c.get_lr()[0], current_epoch)

            # train
            print(f'\n==> training epoch {current_epoch}/{configs.train.num_epochs}')
            train(model, source_loader=source_loaders['train'],
                  target_loader=target_loaders['train'],
                  criterion=criterion, optimizer_g=optimizer_g, optimizer_cls=optimizer_cls,
                  optimizer_dis=optimizer_dis, scheduler_g=scheduler_g, scheduler_cls=scheduler_c,
                  current_step=current_step, writer=writer, cons=cons)
            current_step += step_size

            # evaluate
            meters = dict()
            for split, loader in target_loaders.items():
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
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_cls': optimizer_cls.state_dict(),
                'optimizer_dis': optimizer_dis.state_dict(),
                'meters': meters,
                'configs': configs,
            }, configs.train.checkpoint_path)
            shutil.copyfile(configs.train.checkpoint_path,
                            configs.train.checkpoints_path.format(
                                current_epoch))
            for m in configs.train.metrics:
                if best[m]:
                    shutil.copyfile(configs.train.checkpoint_path,
                                    configs.train.best_checkpoint_paths[m])
            if best.get(configs.train.metric, False):
                shutil.copyfile(configs.train.checkpoint_path,
                                configs.train.best_checkpoint_path)
            print(f'[save_path] = {configs.train.save_path}')


if __name__ == '__main__':
    main()
