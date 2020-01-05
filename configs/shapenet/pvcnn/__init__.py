import torch.optim as optim

from models.shapenet import PVCNN
from utils.config import Config, configs

# model
configs.model = Config(PVCNN)
configs.model.num_classes = configs.data.num_classes
configs.model.num_shapes = configs.data.num_shapes
configs.model.extra_feature_channels = 3

configs.train.num_epochs = 250
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs
