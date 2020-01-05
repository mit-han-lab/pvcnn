# Copyright 2018 Charles R. Qi from Stanford University and
# Wei Liu from Nuro Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numba
import numpy as np
from scipy.spatial import ConvexHull

__all__ = ['get_box_iou_3d']


@numba.njit()
def poly_area(coords):
    """
    calculate area of polygon given x-y coordinates
    (ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates)
    :param coords: FloatTensor[4, 2]
    """
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_clip(subject_polygon, clip_polygon):
    """
    clip a polygon with another polygon
    (ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python)
    :param subject_polygon: a list of (x,y) 2d points, any polygon
    :param clip_polygon: a list of (x,y) 2d points, has to be *convex*
    :return:
        a list of (x,y) vertex point for the intersection polygon
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def compute_intersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    output_list = subject_polygon
    cp1 = clip_polygon[-1]

    for clip_vertex in clip_polygon:
        cp2 = clip_vertex
        input_list = output_list
        output_list = []
        s = input_list[-1]

        for subject_vertex in input_list:
            e = subject_vertex
            if inside(e):
                if not inside(s):
                    output_list.append(compute_intersection())
                output_list.append(e)
            elif inside(s):
                output_list.append(compute_intersection())
            s = e
        cp1 = cp2
        if len(output_list) == 0:
            return None
    return output_list


def convex_hull_intersection(p1, pt):
    """
    compute area of two convex hull's intersection area
    :param p1: a list of (x,y) tuples of hull vertices
    :param pt: a list of (x,y) tuples of hull vertices
    :return:
        a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, pt)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


@numba.njit()
def box_volume_3d(corners):
    a = np.sqrt(np.sum((corners[:, 0] - corners[:, 1]) ** 2))
    b = np.sqrt(np.sum((corners[:, 1] - corners[:, 2]) ** 2))
    c = np.sqrt(np.sum((corners[:, 0] - corners[:, 4]) ** 2))
    return a * b * c


def get_box_iou_3d(corners_1, corners_t):
    """
    calculate iou of 3d box
    :param corners_1: FloatTensor[B, 3, 8], assume up direction is negative Y
    :param corners_t: FloatTensor[B, 3, 8], assume up direction is negative Y
        NOTE: corner points are in counter clockwise order, e.g.,
          2--1
        3--0 5
        7--4
    :return:
        iou_3d: 3D bounding box IoU, FloatTensor[B]
        iou_2d: bird's eye view 2D bounding box IoU, FloatTensor[B]
    """
    if corners_1.ndim == 3:
        batch_size = corners_1.shape[0]
        iou_3d = np.zeros(batch_size)
        iou_2d = np.zeros(batch_size)
        for b in range(batch_size):
            iou_3d[b], iou_2d[b] = get_box_iou_3d(corners_1[b], corners_t[b])
        return iou_3d, iou_2d
    else:
        # corner points are in counter clockwise order
        corners_1_upper_xz = [(corners_1[0, 3], corners_1[2, 3]), (corners_1[0, 2], corners_1[2, 2]),
                              (corners_1[0, 1], corners_1[2, 1]), (corners_1[0, 0], corners_1[2, 0])]
        corners_t_upper_xz = [(corners_t[0, 3], corners_t[2, 3]), (corners_t[0, 2], corners_t[2, 2]),
                              (corners_t[0, 1], corners_t[2, 1]), (corners_t[0, 0], corners_t[2, 0])]
        area_1 = poly_area(np.array(corners_1_upper_xz))
        area_2 = poly_area(np.array(corners_t_upper_xz))
        inter, inter_area = convex_hull_intersection(corners_1_upper_xz, corners_t_upper_xz)
        iou_2d = inter_area / (area_1 + area_2 - inter_area)
        y_max = min(corners_1[1, 0], corners_t[1, 0])
        y_min = max(corners_1[1, 4], corners_t[1, 4])
        inter_vol = inter_area * max(0.0, y_max - y_min)
        vol1 = box_volume_3d(corners_1)
        vol2 = box_volume_3d(corners_t)
        iou_3d = inter_vol / (vol1 + vol2 - inter_vol)
        return iou_3d, iou_2d
