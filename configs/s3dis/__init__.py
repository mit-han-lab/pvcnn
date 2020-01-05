import torch.nn as nn
import torch.optim as optim

from datasets.s3dis import S3DIS
from meters.s3dis import MeterS3DIS
from evaluate.s3dis.eval import evaluate
from utils.config import Config, configs

configs.data.num_classes = 13

# dataset configs
configs.dataset = Config(S3DIS)
configs.dataset.root = 'data/s3dis/pointcnn'
configs.dataset.with_normalized_coords = True
# configs.dataset.num_points = 2048
# configs.dataset.holdout_area = 5

# evaluate configs
configs.evaluate = Config()
configs.evaluate.fn = evaluate
configs.evaluate.num_votes = 1
configs.evaluate.batch_size = 10
configs.evaluate.dataset = Config(split='test')

# train configs
configs.train = Config()
configs.train.num_epochs = 50
configs.train.batch_size = 32

# train: meters
configs.train.meters = Config()
configs.train.meters['acc/iou_{}'] = Config(MeterS3DIS, metric='iou', num_classes=configs.data.num_classes)
configs.train.meters['acc/acc_{}'] = Config(MeterS3DIS, metric='overall', num_classes=configs.data.num_classes)

# train: metric for save best checkpoint
configs.train.metric = 'acc/iou_test'

# train: criterion
configs.train.criterion = Config(nn.CrossEntropyLoss)

# train: optimizer
configs.train.optimizer = Config(optim.Adam)
configs.train.optimizer.lr = 1e-3
