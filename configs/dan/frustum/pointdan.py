import torch.optim as optim

from models.frustum_net import FrustumPointDAN
from utils.config import Config, configs

# model
configs.model = Config(FrustumPointDAN)
configs.model.num_classes = configs.data.num_classes
configs.model.num_heading_angle_bins = configs.data.num_heading_angle_bins
configs.model.num_size_templates = configs.data.num_size_templates
configs.model.num_points_per_object = configs.data.num_points_per_object
configs.model.size_templates = configs.data.size_templates
configs.model.extra_feature_channels = 1

# train: scheduler
configs.train.scheduler_g = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler_g.T_max = configs.train.num_epochs
configs.train.scheduler_c = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler_c.T_max = configs.train.num_epochs
