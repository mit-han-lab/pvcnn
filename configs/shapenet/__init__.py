import torch.nn as nn
import torch.optim as optim

from datasets.shapenet import ShapeNet
from meters.shapenet import MeterShapeNet
from evaluate.shapenet.eval import evaluate
from utils.config import Config, configs

configs.data.num_classes = 50
configs.data.num_shapes = 16

# dataset configs
configs.dataset = Config(ShapeNet)
configs.dataset.root = 'data/shapenet/shapenetcore_partanno_segmentation_benchmark_v0_normal'
configs.dataset.with_normal = True
configs.dataset.with_one_hot_shape_id = True
configs.dataset.normalize = True
configs.dataset.jitter = True
configs.dataset.num_points = 2048

# evaluate configs
configs.evaluate = Config()
configs.evaluate.fn = evaluate
configs.evaluate.num_votes = 10
configs.evaluate.dataset = Config(split='test')

# train configs
configs.train = Config()
configs.train.num_epochs = 200
configs.train.batch_size = 32

# train: meters
configs.train.meters = Config()
configs.train.meters['acc/iou_{}'] = Config(MeterShapeNet, num_classes=configs.data.num_classes)

# train: metric for save best checkpoint
configs.train.metric = 'acc/iou_test'

# train: criterion
configs.train.criterion = Config(nn.CrossEntropyLoss)

# train: optimizer
configs.train.optimizer = Config(optim.Adam)
configs.train.optimizer.lr = 1e-3
