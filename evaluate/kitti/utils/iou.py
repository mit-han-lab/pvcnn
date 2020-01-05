# ref: https://github.com/traveller59/second.pytorch/blob/master/second/core/non_max_suppression/nms_gpu.py

import math

import numba
import numpy as np
from numba import cuda

__all__ = ['rotate_iou_gpu_eval']


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@cuda.jit(device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4,), dtype=numba.float32)
    corners_y = cuda.local.array((4,), dtype=numba.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@cuda.jit(device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]
    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]
    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    ab_ab = ab0 * ab0 + ab1 * ab1
    ab_ap = ab0 * ap0 + ab1 * ap1
    ad_ad = ad0 * ad0 + ad1 * ad1
    ad_ap = ad0 * ap0 + ad1 * ap1

    eps = -1e-6
    return ab_ab - ab_ap >= eps and ab_ap >= eps and ad_ad - ad_ap >= eps and ad_ap >= eps


@cuda.jit(device=True, inline=True)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    a = cuda.local.array((2,), dtype=numba.float32)
    b = cuda.local.array((2,), dtype=numba.float32)
    c = cuda.local.array((2,), dtype=numba.float32)
    d = cuda.local.array((2,), dtype=numba.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]
    ba_0 = b[0] - a[0]
    ba_1 = b[1] - a[1]
    da_0 = d[0] - a[0]
    ca_0 = c[0] - a[0]
    da_1 = d[1] - a[1]
    ca_1 = c[1] - a[1]
    acd = da_1 * ca_0 > ca_1 * da_0
    bcd = (d[1] - b[1]) * (c[0] - b[0]) > (c[1] - b[1]) * (d[0] - b[0])
    if acd != bcd:
        abc = ca_1 * ba_0 > ba_1 * ca_0
        abd = da_1 * ba_0 > ba_1 * da_0
        if abc != abd:
            dc0 = d[0] - c[0]
            dc1 = d[1] - c[1]
            ab_ba = a[0] * b[1] - b[0] * a[1]
            cd_dc = c[0] * d[1] - d[0] * c[1]
            dh = ba_1 * dc0 - ba_0 * dc1
            dx = ab_ba * dc0 - ba_0 * cd_dc
            dy = ab_ba * dc1 - ba_1 * cd_dc
            temp_pts[0] = dx / dh
            temp_pts[1] = dy / dh
            return True
    return False


@cuda.jit(device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2,), dtype=numba.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@cuda.jit(device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = cuda.local.array((2,), dtype=numba.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = cuda.local.array((2,), dtype=numba.float32)
        vs = cuda.local.array((16,), dtype=numba.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1
                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@cuda.jit(device=True, inline=True)
def triangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@cuda.jit(device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(triangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4], int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@cuda.jit(device=True, inline=True)
def inter(rbbox1, rbbox2):
    corners1 = cuda.local.array((8,), dtype=numba.float32)
    corners2 = cuda.local.array((8,), dtype=numba.float32)
    intersection_corners = cuda.local.array((16,), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2, intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    return area(intersection_corners, num_intersection)


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def dev_rotate_iou_eval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


@cuda.jit('(int64, int64, float32[:], float32[:], float32[:], int32)', fastmath=False)
def rotate_iou_kernel_eval(N, K, dev_boxes, dev_query_boxes, dev_iou, criterion=-1):
    threads_per_block = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threads_per_block, threads_per_block)
    col_size = min(K - col_start * threads_per_block, threads_per_block)
    block_boxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)

    dev_query_box_idx = threads_per_block * col_start + tx
    dev_box_idx = threads_per_block * row_start + tx
    if tx < col_size:
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if tx < row_size:
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threads_per_block * K + col_start * threads_per_block + tx * K + i
            dev_iou[offset] = dev_rotate_iou_eval(
                block_qboxes[i * 5:i * 5 + 5], block_boxes[tx * 5:tx * 5 + 5], criterion
            )


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """
    rotated box iou running in gpu. 8x faster than cpu version (take 5ms in one example with numba.cuda code).
    convert from [this project](https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation).
    :param boxes: rbboxes, format: centers, dims, angles(clockwise when positive), FloatTensor[N, 5]
    :param query_boxes: FloatTensor[K, 5]
    :param criterion: optional, default: -1
    :param device_id: int, optional, default: 0
    :return:
    """
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threads_per_block = 8 * 8
    cuda.select_device(device_id)
    blocks_per_grid = (div_up(N, threads_per_block), div_up(K, threads_per_block))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blocks_per_grid, threads_per_block, stream](N, K, boxes_dev, query_boxes_dev,
                                                                           iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)
