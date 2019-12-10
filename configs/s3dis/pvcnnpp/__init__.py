from models.s3dis import PVCNN2
from utils.config import Config, configs

# model
configs.model = Config(PVCNN2)
configs.model.num_classes = configs.data.num_classes
configs.dataset.num_points = 8192
