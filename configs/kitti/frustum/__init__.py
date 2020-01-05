import numpy as np
import torch
import torch.optim as optim

from datasets.kitti import FrustumKitti
from datasets.kitti.attributes import kitti_attributes as kitti
from meters.kitti import MeterFrustumKitti
from modules.frustum import FrustumPointNetLoss
from evaluate.kitti.frustum.eval import evaluate
from utils.config import Config, configs

# data configs
configs.data.num_points_per_object = 512
configs.data.num_heading_angle_bins = 12
configs.data.size_template_names = kitti.class_names
configs.data.num_size_templates = len(configs.data.size_template_names)
configs.data.class_name_to_size_template_id = {
    cat: cls for cls, cat in enumerate(configs.data.size_template_names)
}
configs.data.size_template_id_to_class_name = {
    v: k for k, v in configs.data.class_name_to_size_template_id.items()
}
configs.data.size_templates = np.zeros((configs.data.num_size_templates, 3))
for i in range(configs.data.num_size_templates):
    configs.data.size_templates[i, :] = kitti.class_name_to_size_template[
        configs.data.size_template_id_to_class_name[i]]
configs.data.size_templates = torch.from_numpy(configs.data.size_templates.astype(np.float32))

# dataset configs
configs.dataset = Config(FrustumKitti)
configs.dataset.root = 'data/kitti/frustum/frustum_data'
configs.dataset.num_points = 1024
configs.dataset.classes = configs.data.classes
configs.dataset.num_heading_angle_bins = configs.data.num_heading_angle_bins
configs.dataset.class_name_to_size_template_id = configs.data.class_name_to_size_template_id
configs.dataset.random_flip = True
configs.dataset.random_shift = True
configs.dataset.frustum_rotate = True
configs.dataset.from_rgb_detection = False

# evaluate configs
configs.evaluate.fn = evaluate
configs.evaluate.batch_size = 32
configs.evaluate.dataset = Config(split='val', from_rgb_detection=True)

# train configs
configs.train = Config()
configs.train.num_epochs = 209
configs.train.batch_size = 32

# train: meters
configs.train.meters = Config()
for name, metric in [
    ('acc/iou_3d_{}', 'iou_3d'), ('acc/acc_{}', 'accuracy'),
    ('acc/iou_3d_acc_{}', 'iou_3d_accuracy'), ('acc/iou_3d_class_acc_{}', 'iou_3d_class_accuracy')
]:
    configs.train.meters[name] = Config(
        MeterFrustumKitti, metric=metric, num_heading_angle_bins=configs.data.num_heading_angle_bins,
        num_size_templates=configs.data.num_size_templates, size_templates=configs.data.size_templates,
        class_name_to_class_id={cat: cls for cls, cat in enumerate(configs.data.classes)}
    )

# train: metric for save best checkpoint
configs.train.metrics = ('acc/iou_3d_class_acc_val', 'acc/iou_3d_acc_val')

# train: criterion
configs.train.criterion = Config(FrustumPointNetLoss)
configs.train.criterion.num_heading_angle_bins = configs.data.num_heading_angle_bins
configs.train.criterion.num_size_templates = configs.data.num_size_templates
configs.train.criterion.size_templates = configs.data.size_templates
configs.train.criterion.box_loss_weight = 1.0
configs.train.criterion.corners_loss_weight = 10.0
configs.train.criterion.heading_residual_loss_weight = 20.0
configs.train.criterion.size_residual_loss_weight = 20.0

# train: optimizer
configs.train.optimizer = Config(optim.Adam)
configs.train.optimizer.lr = 1e-3
