# Point-Voxel CNN for Efficient 3D Deep Learning [[Website]](https://hanlab.mit.edu/projects/pvcnn/) [[arXiv]](https://arxiv.org/abs/1907.03739)


```
@inproceedings{liu2019pvcnn,
  title={Point-Voxel CNN for Efficient 3D Deep Learning},
  author={Liu, Zhijian and Tang, Haotian and Lin, Yujun and Han, Song},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

<img src="https://hanlab.mit.edu/projects/pvcnn/figures/gif/PVCNN-livedemo-p1-480p.gif" width="1080"><img src="https://hanlab.mit.edu/projects/pvcnn/figures/gif/PVCNN-livedemo-p2-480p.gif" width="1080"><img src="https://hanlab.mit.edu/projects/pvcnn/figures/gif/PVCNN-livedemo-p3-480p.gif" width="1080">

## Overview

We release the PyTorch code of the [Point-Voxel CNN](https://arxiv.org/abs/1907.03739).
<img src="https://hanlab.mit.edu/projects/pvcnn/figures/overview.png" width="1080">

## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
  * S3DIS
- [Code](#code)
- [Pretrained Models](#pretrained-models)
  * S3DIS
- [Testing Pretrained Models](#testing-pretrained-models)
- [Training](#training)

## Prerequisites

The code is built with following libraries:
- Python >= 3.6
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.0
- [tensorboardX](https://github.com/lanpa/tensorboardX) >= 1.2
- [h5py](https://github.com/h5py/h5py) >= 2.9.0
- [numba](https://github.com/numba/numba)
- [tqdm](https://github.com/tqdm/tqdm)

For point data pre-processing, you may need [plyfile](https://github.com/dranjan/python-plyfile).

## Data Preparation

### S3DIS

We follow the data pre-processing in [PointCNN](https://github.com/yangyanli/PointCNN/).
The code for preprocessing the S3DIS dataset is located in [`scripts/s3dis/`](scripts/s3dis/prepare_data.py).
You should first download the dataset from [here](http://buildingparser.stanford.edu/dataset.html), then run 
```
python scripts/s3dis/prepare_data.py -d [path to unzip dataset dir]
```

## Code

This code is based on [PointCNN](https://github.com/yangyanli/PointCNN/) and [Pointnet2_PyTorch](https://https://github.com/erikwijmans/Pointnet2_PyTorch).
We modified the code for PyTorch-style data layout.
The core code to implement PVConv is [modules/pvconv.py](modules/pvconv.py). Its key idea costs only a few lines of code:

```python
    voxel_features, voxel_coords = voxelize(features, coords)
    voxel_features = voxel_layers(voxel_features)
    voxel_features = trilinear_devoxelize(voxel_features, voxel_coords, resolution)
    fused_features = voxel_features + point_layers(features)
```

## Pretrained Models

Here we provide some of the pretrained models. The accuracy might vary a little bit compared to the paper, since we re-train some of the models for reproducibility.

### S3DIS

We compare the 3D-UNet and PointCNN performance reported in the following table.
The accuracy is tested following [here](https://github.com/yangyanli/PointCNN/). The list is keeping updating.

|                    | Overall Acc |   mIoU   | 
| :----------------: | :---------: | :------: |
|  3D-UNet           |    85.12    |   54.93  |
|  [PVCNN](https://hanlab.mit.edu/projects/pvcnn/files/models/s3dis.pvcnn.area5.pth.tar)     |    86.16    |   56.17  |
|  PointCNN          |    85.91    |   57.26  |
|  [PVCNN++](https://hanlab.mit.edu/projects/pvcnn/files/models/s3dis.pvcnnpp.area5.pth.tar) |    87.14    |   58.33  |


## Testing Pretrained Models

For example, to test the downloaded pretrained models on S3DIS, you can run

```
python train.py [config-file] --devices [gpu-ids] --evaluate --configs.train.best_checkpoint_path [path to your models]
```

For instance, if you want to evaluate PVCNN on GPU 0,1 (with 4096 points on Area 1-4 & 6), you can run

```
python train.py configs/s3dis/pvcnn/area5.py --devices 0,1 --evaluate --configs.train.best_checkpoint_path s3dis.pvcnn.area5.pth.tar
```

## Training

We provided several examples to train PVCNN with this repo:

- To train PVCNN on S3DIS holding out Area 5, you can run
 
```
python train.py configs/s3dis/pvcnn/area5.py --devices 0,1
```

- To train PVCNN++ on S3DIS holding out Area 5, you can run
 
```
python train.py configs/s3dis/pvcnnpp/area5.py --devices 0,1
```

In general, to train a model, you can run
 
```
python train.py [config-file] --devices [gpu-ids]
```

To evaluate trained models, you can do inference by running:

```
python train.py [config-file] --devices [gpu-ids] --evaluate
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
