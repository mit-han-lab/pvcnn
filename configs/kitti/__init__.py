from utils.config import Config, configs

# data configs
configs.data.classes = ('Car', 'Pedestrian', 'Cyclist')
configs.data.num_classes = len(configs.data.classes)

# evaluate configs
configs.evaluate = Config()
configs.evaluate.num_tests = 20
configs.evaluate.ground_truth_path = 'data/kitti/ground_truth'
configs.evaluate.image_id_file_path = 'data/kitti/image_sets/val.txt'
