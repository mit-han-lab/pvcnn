import torch

from models.point_dan.model_utils import *
from torch import nn

# Channel Attention
from models.utils import create_mlp_components, create_pointnet_components


class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(4096)

    def forward(self, x):
        y = self.conv_du(x)
        y = x * y + x
        y = y.view(y.shape[0], -1)
        y = self.bn(y)

        return y


# Grad Reversal
class GradReverse(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -1.0


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x)


# Generator
class PointnetG(nn.Module):
    def __init__(self, in_channels):
        super(PointnetG, self).__init__()
        self.in_channels= in_channels
        self.trans_net1 = transform_net(4, 4)
        self.trans_net2 = transform_net(64, 64)

        point_blocks = ((64, 3, None),)
        cloud_blocks = ((128, 1, None), (1024, 1, None))

        layers, channels_point, _ = create_pointnet_components(
            blocks=point_blocks, in_channels=self.in_channels, with_se=False
        )
        self.point_features = nn.Sequential(*layers)

        # SA Node Module
        self.conv3 = adapt_layer_off()  # (64->128)
        layers, channels_cloud, _ = create_pointnet_components(
            blocks=cloud_blocks, in_channels=2*channels_point, with_se=False,
        )
        self.cloud_features = nn.Sequential(*layers)

    def forward(self, x, node=False):
        x_loc = x.squeeze(-1)

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        
        x = x.transpose(2, 1)
        x = self.point_features(x)
        x = x.unsqueeze(-1)
        transform = self.trans_net2(x)
        
        x = x.transpose(2, 1)
        x = x.squeeze(-1)
        x = torch.bmm(x, transform)

        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x, node_feat, node_off = self.conv3(x, x_loc)
        
        x = x.squeeze(-1)
        point_feat = x

        x = self.cloud_features(x)
        x, _ = torch.max(x, dim=-1, keepdim=True)

        cloud_feat = x

        if node:
            return cloud_feat, point_feat, node_feat, node_off
        else:
            return cloud_feat, point_feat, node_feat


class InstanceSegmentationPointDAN(nn.Module):

    def __init__(self, num_classes=3, extra_feature_channels=1, width_multiplier=1):
        super(InstanceSegmentationPointDAN, self).__init__()
        self.in_channels = extra_feature_channels + 3
        self.num_classes = num_classes

        self.g = PointnetG(self.in_channels)

        self.attention_s = CALayer(64 * 64)
        self.attention_t = CALayer(64 * 64)

        channels_point = 128
        channels_cloud = 1024

        layers, _ = create_mlp_components(
            in_channels=(channels_point + channels_cloud + self.num_classes),
            out_channels=[512, 256, 128, 128, 0.5, 2],
            classifier=True, dim=2, width_multiplier=1
        )
        self.c1 = nn.Sequential(*layers)

        layers, _ = create_mlp_components(
            in_channels=( channels_point + channels_cloud + self.num_classes),
            out_channels=[512, 256, 128, 128, 0.5, 2],
            classifier=True, dim=2, width_multiplier=1
        )
        self.c2 = nn.Sequential(*layers)

    def forward(self, inputs, constant=1, adaptation=False, node_vis=False,
                node_adaptation_s=False, node_adaptation_t=False):

        features = inputs['features']
        num_points = features.size(-1)
        one_hot_vectors = inputs['one_hot_vectors'].unsqueeze(-1).repeat(
            [1, 1, num_points])

        assert one_hot_vectors.dim() == 3  # [B, C, N]
        
        features = features.unsqueeze(-1)
        cloud_feat, point_feat, feat_ori, node_idx = self.g(features, node=True)
        batch_size = feat_ori.size(0)

        # sa node visualization
        if node_vis:
            return node_idx

        if node_adaptation_s:
            # source domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_s = self.attention_s(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_s

        elif node_adaptation_t:
            # target domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_t = self.attention_t(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_t

        if adaptation:
            cloud_feat = grad_reverse(cloud_feat, constant)

        cloud_feat = cloud_feat.repeat([1, 1, num_points]) 
        cls_input = torch.cat([one_hot_vectors, point_feat, cloud_feat], dim=1)

        y1 = self.c1(cls_input)
        y2 = self.c2(cls_input)

        return y1, y2
