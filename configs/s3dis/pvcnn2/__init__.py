import torch.optim as optim

from models.s3dis import PVCNN2
from utils.config import Config, configs

# model
configs.model = Config(PVCNN2)
configs.model.num_classes = configs.data.num_classes
configs.model.extra_feature_channels = 6
configs.dataset.num_points = 8192

configs.train.optimizer.weight_decay = 1e-5
# train: scheduler
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs
