import torch
import torch.nn as nn


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, :, farthest].view(B, 4, 1)
        dist = torch.sum((xyz - centroid) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, C, N]/[B,C,N,1]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, C, S]
    """
    if len(points.shape) == 4:
        points = points.squeeze()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    points = points.permute(0,2,1) #(B,N,C)
    new_points = points[batch_indices, idx, :]
    if len(new_points.shape)==3:
        new_points = new_points.permute(0,2,1)
    elif len(new_points.shape) == 4:
        new_points = new_points.permute(0,3,1,2)
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, C, N]
        new_xyz: query points, [B, C, S]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, C, N = xyz.shape
    _, _, S = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)
    if radius is not None:
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
    else:
        group_idx = torch.sort(sqrdists, dim=-1)[1][:,:,:nsample]
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, C, N]
        dst: target points, [B, C, M]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, _, N = src.shape
    _, _, M = dst.shape
    dist = -2 * torch.matmul(src.permute(0, 2, 1), dst)
    dist += torch.sum(src ** 2, 1).view(B, N, 1)
    dist += torch.sum(dst ** 2, 1).view(B, 1, M)
    return dist

def upsample_inter(xyz1, xyz2, points1, points2, k):
    """
    Input:
        xyz1: input points position data, [B, C, N]
        xyz2: sampled input points position data, [B, C, S]
        points1: input points data, [B, D, N]/[B,D,N,1]
        points2: input points data, [B, D, S]/[B,D,S,1]
        k:
    Return:
        new_points: upsampled points data, [B, D+D, N]
    """
    if points1 is not None:
        if len(points1.shape) == 4:
            points1 = points1.squeeze()
    if len(points2.shape) == 4:
        points2 = points2.squeeze()
    B, C, N = xyz1.size()
    _, _, S = xyz2.size()

    dists = square_distance(xyz1, xyz2) #(B, N, S)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :k], idx[:, :, :k]  # [B, N, 3]
    dists[dists < 1e-10] = 1e-10
    weight = 1.0 / dists  # [B, N, 3]
    weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]; weight = [64, 1024, 3]
    interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, 1, N, k), dim=3) #(B,D,N); idx = [64, 1024, 3]; points2 = [64, 64, 64];
    if points1 is not None:
        new_points = torch.cat([points1, interpolated_points], dim=1)  # points1 = [64, 64, 1024];
        return new_points
    else:
        return interpolated_points



def pairwise_distance(x):
    batch_size = x.size(0)
    point_cloud = torch.squeeze(x)
    if batch_size == 1:
        point_cloud = torch.unsqueeze(point_cloud, 0)
    point_cloud_transpose = torch.transpose(point_cloud, dim0=1, dim1=2)
    point_cloud_inner = torch.matmul(point_cloud_transpose, point_cloud)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(point_cloud ** 2, dim=1, keepdim=True)
    point_cloud_square_transpose = torch.transpose(point_cloud_square, dim0=1, dim1=2)
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose


def gather_neighbor(x, nn_idx, n_neighbor):
    x = torch.squeeze(x)
    batch_size = x.size()[0]
    num_dim = x.size()[1]
    num_point = x.size()[2]
    point_expand = x.unsqueeze(2).expand(batch_size, num_dim, num_point, num_point)
    nn_idx_expand = nn_idx.unsqueeze(1).expand(batch_size, num_dim, num_point, n_neighbor)
    pc_n = torch.gather(point_expand, -1, nn_idx_expand)
    return pc_n

def get_neighbor_feature(x, n_point, n_neighbor):
    if len(x.size()) == 3:
        x = x.unsqueeze()
    adj_matrix = pairwise_distance(x)
    _, nn_idx = torch.topk(adj_matrix, n_neighbor, dim=2, largest=False)
    nn_idx = nn_idx[:, :n_point, :]
    batch_size = x.size()[0]
    num_dim = x.size()[1]
    num_point = x.size()[2]
    point_expand = x[:, :, :n_point, :].expand(-1, -1, -1, num_point)
    nn_idx_expand = nn_idx.unsqueeze(1).expand(batch_size, num_dim, n_point, n_neighbor)
    pc_n = torch.gather(point_expand, -1, nn_idx_expand)
    return pc_n


def get_edge_feature(x, n_neighbor):
    if len(x.size()) == 3:
        x = x.unsqueeze(3)
    adj_matrix = pairwise_distance(x)
    _, nn_idx = torch.topk(adj_matrix, n_neighbor, dim=2, largest=False)
    point_cloud_neighbors = gather_neighbor(x, nn_idx, n_neighbor)
    point_cloud_center = x.expand(-1, -1, -1, n_neighbor)
    edge_feature = torch.cat((point_cloud_center, point_cloud_neighbors-point_cloud_center), dim=1)
    return edge_feature

