# PVCNN: Point-Voxel CNN for Efficient 3D Deep Learning

[NVIDIA Jetson Community Project Spotlight](https://news.developer.nvidia.com/point-voxel-cnn-3d/?ncid=so-twit-99540#cid=em02_so-twit_en-us)

```
@inproceedings{liu2019pvcnn,
  title={Point-Voxel CNN for Efficient 3D Deep Learning},
  author={Liu, Zhijian and Tang, Haotian and Lin, Yujun and Han, Song},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

## Prerequisites

The code is built with following libraries (see [requirements.txt](requirements.txt)):
- Python >= 3.7
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.3
- [numba](https://github.com/numba/numba)
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [six](https://github.com/benjaminp/six)
- [tensorboardX](https://github.com/lanpa/tensorboardX) >= 1.2
- [tqdm](https://github.com/tqdm/tqdm)
- [plyfile](https://github.com/dranjan/python-plyfile)
- [h5py](https://github.com/h5py/h5py)

## Data Preparation

### S3DIS

We follow the data pre-processing in [PointCNN](https://github.com/yangyanli/PointCNN).
The code for preprocessing the S3DIS dataset is located in [`data/s3dis/`](data/s3dis/prepare_data.py).
One should first download the dataset from [here](http://buildingparser.stanford.edu/dataset.html), then run 
```bash
python data/s3dis/prepare_data.py -d [path to unzipped dataset dir]
```

### ShapeNet

We follow the data pre-processing in [PointNet2](https://github.com/charlesq34/pointnet2). Please run the following
command to down the dataset
```bash
./data/shapenet/download.sh
```

### KITTI

For Frustum-PointNet backbone, we follow the data pre-processing in [Frustum-Pointnets](https://github.com/charlesq34/frustum-pointnets).
One should first download the ground truth labels from [here](http://www.cvlibs.net/download.php?file=data_object_label_2.zip), then run
```bash
unzip data_object_label_2.zip
mv training/label_2 data/kitti/ground_truth
./data/kitti/frustum/download.sh
```

## Code

The core code to implement PVConv is [modules/pvconv.py](modules/pvconv.py). Its key idea costs only a few lines of code:

```python
    voxel_features, voxel_coords = voxelize(features, coords)
    voxel_features = voxel_layers(voxel_features)
    voxel_features = trilinear_devoxelize(voxel_features, voxel_coords, resolution)
    fused_features = voxel_features + point_layers(features)
```

## Pretrained Models

Here we provide some of the pretrained models. The accuracy might vary a little bit compared to the paper,
since we re-train some of the models for reproducibility.

### S3DIS

We compare PVCNN against the PointNet, 3D-UNet and PointCNN performance as reported in the following table.
The accuracy is tested following [PointCNN](https://github.com/yangyanli/PointCNN). The list is keeping updated.

|                                                  Models                                                     | Overall Acc |     mIoU     | 
| :---------------------------------------------------------------------------------------------------------: | :---------: | :----------: |
|  PointNet                                                                                                   |    82.54    |     42.97    |
|  [PointNet (Reproduce)](https://hanlab.mit.edu/files/pvcnn/s3dis.pointnet.area5.pth.tar)    |    80.46    |     44.03    |
|  [PVCNN (0.125 x C)](https://hanlab.mit.edu/files/pvcnn/s3dis.pvcnn.area5.c0p125.pth.tar)   |    82.79    |   **48.75**  |
|  [PVCNN (0.25  x C)](https://hanlab.mit.edu/files/pvcnn/s3dis.pvcnn.area5.c0p25.pth.tar)    |    85.00    |   **53.08**  |
|  3D-UNet                                                                                                    |    85.12    |     54.93    |
|  [PVCNN](https://hanlab.mit.edu/files/pvcnn/s3dis.pvcnn.area5.c1.pth.tar)                   |    86.47    |   **56.64**  |
|  PointCNN                                                                                                   |    85.91    |     57.26    |
|  [PVCNN++ (0.5 x C)](https://hanlab.mit.edu/files/pvcnn/s3dis.pvcnn2.area5.c0p5.pth.tar)    |    86.88    |   **58.30**  |
|  [PVCNN++](https://hanlab.mit.edu/files/pvcnn/s3dis.pvcnn2.area5.c1.pth.tar)                |    87.48    |   **59.02**  |

### ShapeNet
We compare PVCNN against the PointNet, PointNet++, 3D-UNet, Spider CNN and PointCNN performance as reported in the following table.
The accuracy is tested following [PointNet](https://github.com/charlesq34/pointnet2). The list is keeping updated.

|                                                  Models                                                         |     mIoU     | 
| :-------------------------------------------------------------------------------------------------------------: | :----------: |
|  [PointNet (Reproduce)](https://hanlab.mit.edu/files/pvcnn/shapenet.pointnet.pth.tar)           |     83.5     |
|  PointNet                                                                                                       |     83.7     |
|  3D-UNet                                                                                                        |     84.6     |
|  [PVCNN (0.25 x C)](https://hanlab.mit.edu/files/pvcnn/shapenet.pvcnn.c0p25.pth.tar)            |   **84.9**   |
|  [PointNet++ SSG (Reproduce)](https://hanlab.mit.edu/files/pvcnn/shapenet.pointnet2ssg.pth.tar) |     85.1     |
|  PointNet++ MSG                                                                                                 |     85.1     |
|  [PVCNN (0.25 x C, DML)](https://hanlab.mit.edu/files/pvcnn/shapenet.pvcnn.c0p25.dml.pth.tar)   |   **85.1**   |
|  SpiderCNN                                                                                                      |     85.3     |
|  [PointNet++ MSG (Reproduce)](https://hanlab.mit.edu/files/pvcnn/shapenet.pointnet2msg.pth.tar) |     85.3     |
|  [PVCNN (0.5  x C)](https://hanlab.mit.edu/files/pvcnn/shapenet.pvcnn.c0p5.pth.tar)             |   **85.5**   |
|  [PVCNN](https://hanlab.mit.edu/files/pvcnn/shapenet.pvcnn.c1.pth.tar)                          |   **85.8**   |
|  PointCNN                                                                                                       |     86.1     |
|  [PVCNN (DML)](https://hanlab.mit.edu/files/pvcnn/shapenet.pvcnn.c1.dml.pth.tar)                |   **86.1**   |


### KITTI
We compare PVCNN (Efficient Version in the paper) against PointNets performance as reported in the following table.
The accuracy is tested on **val** set following [Frustum PointNets](https://github.com/charlesq34/frustum-pointnets) using 
modified code from [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python).
Since there is random sampling in Frustum Pointnets, random seed will influence the evaluation. All results provided by us
are the average of 20 measurements with different seeds, and the best one of 20 measurements is shown in the parentheses.
The list is keeping updated.

|                                                   Models                                                             |        Car        |        Car        |        Car        |     Pedestrian    |     Pedestrian    |     Pedestrian    |      Cyclist      |      Cyclist      |      Cyclist      |
|:--------------------------------------------------------------------------------------------------------------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
|                                                                                                                      |        Easy       |      Moderate     |        Hard       |        Easy       |      Moderate     |        Hard       |        Easy       |      Moderate     |        Hard       |
| Frustum PointNet                                                                                                     |       83.26       |       69.28       |       62.56       |         -         |         -         |         -         |         -         |         -         |         -         |
| [Frustum PointNet (Reproduce)](https://hanlab.mit.edu/files/pvcnn/kitti.frustum.pointnet.pth.tar)    |   85.24 (85.17)   |   71.63 (71.56)   |   63.79 (63.78)   |   66.44 (66.83)   |   56.90 (57.20)   |   50.43 (50.54)   |   77.14 (78.16)   |   56.46 (57.41)   |   52.79 (53.66)   |
| Frustum PointNet++                                                                                                   |       83.76       |       70.92       |       63.65       |       70.00       |       61.32       |       53.59       |       77.15       |       56.49       |       53.37       |
| [Frustum PointNet++ (Reproduce)](https://hanlab.mit.edu/files/pvcnn/kitti.frustum.pointnet2.pth.tar) |   84.72 (84.46)   |   71.99 (71.95)   |   64.20 (64.13)   |   68.40 (69.27)   |   60.03 (60.80)   |   52.61 (53.19)   |   75.56 (79.41)   |   56.74 (58.65)   | 53.33 (**54.82**) |
| [Frustum PVCNN (Efficient)](https://hanlab.mit.edu/files/pvcnn/kitti.frustum.pvcnne.pth.tar)         | **85.25 (85.30)** | **72.12 (72.22)** | **64.24 (64.36)** | **70.60 (70.60)** | **61.24 (61.35)** | **56.25 (56.38)** | **78.10 (79.79)** | **57.45 (58.72)** | **53.65 (54.81)** |


## Testing Pretrained Models

For example, to test the downloaded pretrained models on S3DIS, one can run

```
python train.py [config-file] --devices [gpu-ids] --evaluate --configs.evaluate.best_checkpoint_path [path to the model checkpoint]
```

For instance, to evaluate PVCNN on GPU 0,1 (with 4096 points on Area 5 of S3DIS), one can run

```
python train.py configs/s3dis/pvcnn/area5.py --devices 0,1 --evaluate --configs.evaluate.best_checkpoint_path s3dis.pvcnn.area5.c1.pth.tar
```

Specially, for Frustum KITTI evaluation, one can specify the number of measurements to eliminate the random seed effects,

```
python train.py configs/kitti/frustum/pvcnne.py --devices 0 --evaluate --configs.evaluate.best_checkpoint_path kitti.frustum.pvcnne.pth.tar --configs.evaluate.num_tests [#measurements]
```

## Training

We provided several examples to train PVCNN with this repo:

- To train PVCNN on S3DIS holding out Area 5, one can run
 
```
python train.py configs/s3dis/pvcnn/area5/c1.py --devices 0,1
```

In general, to train a model, one can run
 
```
python train.py [config-file] --devices [gpu-ids]
```

**NOTE**: During training, the meters will provide accuracies and IoUs.
However, these are just rough estimations.
One have to run the following command to get accurate evaluation.

To evaluate trained models, one can do inference by running:

```
python train.py [config-file] --devices [gpu-ids] --evaluate
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.


## Acknowledgement

- The code for data [pre-processing](data/s3dis/prepare_data.py) and [evaluation](evaluate/s3dis/eval.py) of S3DIS dataset is modified from [PointCNN](https://github.com/yangyanli/PointCNN/) (MIT License).

- The code for PointNet and PointNet++ [primitive](modules/functional/src) is modified from [PointNet2](https://github.com/charlesq34/pointnet2) (MIT License) and [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
We modified the data layout and merged kernels to speed up and meet with PyTorch style.

- The code for data [pre-processing](datasets/kitti/frustum.py) and [evaluation](meters/kitti/utils.py) of KITTI dataset is modified from [Frustum-Pointnets](https://github.com/charlesq34/frustum-pointnets) (Apache 2.0 License).

- The code for [evaluation](evaluate/kitti/utils) of KITTI dataset is modified from [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python) (MIT License).
