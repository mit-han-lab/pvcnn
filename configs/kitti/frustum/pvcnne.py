import torch.optim as optim

from models.kitti.frustum import FrustumPVCNNE
from utils.config import Config, configs

# model
configs.model = Config(FrustumPVCNNE)
configs.model.num_classes = configs.data.num_classes
configs.model.num_heading_angle_bins = configs.data.num_heading_angle_bins
configs.model.num_size_templates = configs.data.num_size_templates
configs.model.num_points_per_object = configs.data.num_points_per_object
configs.model.size_templates = configs.data.size_templates
configs.model.extra_feature_channels = 1

# train: scheduler
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs
