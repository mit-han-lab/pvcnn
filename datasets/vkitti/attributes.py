import numpy as np

from utils.container import G

__all__ = ['vkitti_attributes']


vkitti_attributes = G()
vkitti_attributes.class_names = ('Car', 'Van', 'Truck')
# ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
vkitti_attributes.class_name_to_size_template = {
    'Car': np.array([3.85150801, 1.59570698, 1.50239394]),
    'Van': np.array([4.66256793, 1.89853565, 2.02366792]),
    'Truck': np.array([9.72038953, 2.67032881, 3.54183852])
}
