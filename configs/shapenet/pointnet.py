import torch.optim as optim

from models.shapenet import PointNet
from utils.config import Config, configs

# model
configs.model = Config(PointNet)
configs.model.num_classes = configs.data.num_classes
configs.model.num_shapes = configs.data.num_shapes
configs.model.extra_feature_channels = 0

configs.dataset.with_normal = False
configs.train.scheduler = Config(optim.lr_scheduler.StepLR)
configs.train.scheduler.step_size = 20
configs.train.scheduler.gamma = 0.5
