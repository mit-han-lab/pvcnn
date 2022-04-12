import torch.optim as optim

from models.s3dis import PVCNN2
from utils.config import Config, configs

# model
configs.model = Config(PVCNN2)
configs.model.num_classes = configs.data.num_classes
configs.model.extra_feature_channels = 6
configs.dataset.num_points = 2048


# train: scheduler
configs.train.num_epochs = 250
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs
