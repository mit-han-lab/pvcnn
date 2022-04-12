import torch.nn as nn
import torch.optim as optim

from datasets.custom_dataset import Custom
from meters.s3dis import MeterS3DIS
from evaluate.s3dis.eval import evaluate
from utils.config import Config, configs

configs.data.num_classes = 10

# dataset configs
configs.dataset = Config(Custom)
configs.dataset.root = 'Path_To_Custom_Dataset'
configs.dataset.with_normal = True
configs.dataset.normalize = True
configs.dataset.jitter = True
configs.dataset.num_points = 2048
configs.dataset.data_aug = True

# evaluate configs
configs.evaluate = Config()
configs.evaluate.fn = evaluate
configs.evaluate.num_votes = 10
configs.evaluate.dataset = Config(split='test')

# train configs
configs.train = Config()
configs.train.num_epochs = 200
configs.train.batch_size = 8

# train: meters
configs.train.meters = Config()
configs.train.meters['acc/acc_{}'] = Config(MeterS3DIS, metric='overall', num_classes=configs.data.num_classes)

# train: metric for save best checkpoint
configs.train.metric = 'acc/acc_test'

# train: criterion
configs.train.criterion = Config(nn.CrossEntropyLoss)

# train: optimizer
configs.train.optimizer = Config(optim.AdamW)
configs.train.optimizer.lr = 0.001
configs.train.optimizer.weight_decay = 0.0005
