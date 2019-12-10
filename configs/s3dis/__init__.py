import torch.nn as nn
import torch.optim as optim

from datasets.s3dis import S3DIS
from meters.s3dis import MeterS3DIS
from scripts.s3dis.eval import evaluate
from utils.config import Config, configs

configs.data.num_classes = 13
configs.data.num_votes = 1

# dataset configs
configs.dataset = Config(S3DIS)
configs.dataset.root = 'data/s3dis/'
configs.dataset.with_normalized_coords = True
# configs.dataset.num_points = 2048
# configs.dataset.holdout_area = 5

# evaluate script
configs.evaluate = evaluate

# train configs
configs.train = Config()
configs.train.num_epochs = 50
configs.train.batch_size = 32

# train: meters
configs.train.meters = Config()
configs.train.meters['acc/iou_{}'] = Config(MeterS3DIS, metric='iou')
configs.train.meters['acc/acc_{}'] = Config(MeterS3DIS, metric='overall')

# train: metric for save best checkpoint
configs.train.metric = 'acc/iou_test'

# train: criterion
configs.train.criterion = Config(nn.CrossEntropyLoss)

# train: optimizer
configs.train.optimizer = Config(optim.Adam)
configs.train.optimizer.lr = 1e-3
configs.train.optimizer.weight_decay = 1e-5

# train: scheduler
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs
