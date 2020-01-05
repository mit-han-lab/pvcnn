# ref: https://github.com/traveller59/kitti-object-eval-python/blob/master/eval.py

import io as sysio

import numba
import numpy as np

from .iou import rotate_iou_gpu_eval


__all__ = ['get_official_eval_result']


def get_map(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = (query_boxes[k, 2] - query_boxes[k, 0]) * (query_boxes[k, 3] - query_boxes[k, 1])
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih
                    elif criterion == 0:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    return rotate_iou_gpu_eval(boxes, qboxes, criterion)


@numba.jit(nopython=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1, z_axis=1, z_center=1.0):
    """
    :param boxes:
    :param qboxes:
    :param rinc:
    :param criterion:
    :param z_axis: the z (height) axis
    :param z_center: unified z (height) center of box
    :return:
    """
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                min_z = min(boxes[i, z_axis] + boxes[i, z_axis + 3] * (1 - z_center),
                            qboxes[j, z_axis] + qboxes[j, z_axis + 3] * (1 - z_center))
                max_z = max(boxes[i, z_axis] - boxes[i, z_axis + 3] * z_center,
                            qboxes[j, z_axis] - qboxes[j, z_axis + 3] * z_center)
                iw = min_z - max_z
                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1, z_axis=1, z_center=1.0):
    """
    kitti camera format z_axis=1.
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)
    rinc = rotate_iou_gpu_eval(boxes[:, bev_axes], qboxes[:, bev_axes], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion, z_axis, z_center)
    return rinc


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50, z_axis=1, z_center=1.0):
    """
    fast iou algorithm. this function can be used independently to do result analysis.
    :param gt_annos: must from get_label_annos() in kitti_common.py, dict
    :param dt_annos: must from get_label_annos() in kitti_common.py, dict
    :param metric: eval type, 0: bbox, 1: bev, 2: 3d
    :param num_parts: a parameter for fast calculate algorithm, int
    :param z_axis: height axis, kitti camera use 1, lidar use 2.
    :param z_center:
    :return:
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    bev_axes = list(range(3))
    bev_axes.pop(z_axis)
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a['bbox'] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a['bbox'] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate([a['location'][:, bev_axes] for a in gt_annos_part], 0)
            dims = np.concatenate([a['dimensions'][:, bev_axes] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a['location'][:, bev_axes] for a in dt_annos_part], 0)
            dims = np.concatenate([a['dimensions'][:, bev_axes] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a['location'] for a in gt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a['location'] for a in dt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes, z_axis=z_axis, z_center=z_center).astype(np.float64)
        else:
            raise ValueError('unknown metric')
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num, dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    _class_names = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'car', 'tractor', 'trailer']
    _min_height = [40, 25, 25]
    _max_occlusion = [0, 1, 2]
    _max_truncation = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = _class_names[current_class].lower()
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno['bbox'][i]
        gt_name = gt_anno['name'][i].lower()
        height = bbox[3] - bbox[1]
        if gt_name == current_cls_name:
            valid_class = 1
        elif current_cls_name == 'Pedestrian'.lower() and 'Person_sitting'.lower() == gt_name:
            valid_class = 0
        elif current_cls_name == 'Car'.lower() and 'Van'.lower() == gt_name:
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno['occluded'][i] > _max_occlusion[difficulty])
                or (gt_anno['truncated'][i] > _max_truncation[difficulty])
                or (height <= _min_height[difficulty])):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore and (valid_class == 1)):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        if gt_anno['name'][i] == 'DontCare':
            dc_bboxes.append(gt_anno['bbox'][i])
    for i in range(num_dt):
        if dt_anno['name'][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno['bbox'][i, 3] - dt_anno['bbox'][i, 1])
        if height < _min_height[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate([gt_annos[i]['bbox'], gt_annos[i]['alpha'][..., np.newaxis]], 1)
        dt_datas = np.concatenate([dt_annos[i]['bbox'], dt_annos[i]['alpha'][..., np.newaxis],
                                   dt_annos[i]['score'][..., np.newaxis]], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares, total_dc_num, total_num_valid_gt


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps, gt_datas, dt_datas, ignored_gt, ignored_det, dc_bboxes, metric, min_overlap,
                           thresh=0, compute_fp=False, compute_aos=False):
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    _no_detection = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = _no_detection
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if ignored_det[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if not compute_fp and (overlap > min_overlap) and dt_score > valid_detection:
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap) and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif compute_fp and (overlap > min_overlap) and (valid_detection == _no_detection) and ignored_det[j] == 1:
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == _no_detection) and ignored_gt[i] == 0:
            fn += 1
        elif (valid_detection != _no_detection) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1):
            assigned_detection[det_idx] = True
        elif valid_detection != _no_detection:
            # only a tp add a threshold.
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if not (assigned_detection[i] or ignored_det[i] == -1 or ignored_det[i] == 1 or ignored_threshold[i]):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if assigned_detection[j]:
                        continue
                    if ignored_det[j] == -1 or ignored_det[j] == 1:
                        continue
                    if ignored_threshold[j]:
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps, pr, gt_nums, dt_nums, dc_nums, gt_datas, dt_datas, dontcares,
                             ignored_gts, ignored_dets, metric, min_overlap, thresholds, compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num + gt_nums[i]]
            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap, gt_data, dt_data, ignored_gt, ignored_det, dontcare, metric,
                min_overlap=min_overlap, thresh=thresh, compute_fp=True, compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def eval_class(gt_annos, dt_annos, current_classes, difficulties, metric, min_overlaps, compute_aos=False, z_axis=1,
               z_center=1.0, num_parts=50):
    """
    Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    :param gt_annos: must from get_label_annos() in kitti_common.py, dict
    :param dt_annos: must from get_label_annos() in kitti_common.py, dict
    :param current_classes: 0: car, 1: pedestrian, 2: cyclist, int
    :param difficulties: eval difficulty, 0: easy, 1: normal, 2: hard, int
    :param metric: eval type, 0: bbox, 1: bev, 2: 3d, int
    :param min_overlaps: [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]] format: [metric, class]
                         choose one from matrix above, float
    :param compute_aos:
    :param z_axis:
    :param z_center:
    :param num_parts:
    :return: dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts, z_axis=z_axis, z_center=z_center)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    _n_sample_pts = 41
    num_min_overlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficulties)
    precision = np.zeros([num_class, num_difficulty, num_min_overlap, _n_sample_pts])
    aos = np.zeros([num_class, num_difficulty, num_min_overlap, _n_sample_pts])
    all_thresholds = np.zeros([num_class, num_difficulty, num_min_overlap, _n_sample_pts])
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficulties):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(overlaps[i], gt_datas_list[i], dt_datas_list[i], ignored_gts[i],
                                                  ignored_dets[i], dontcares[i], metric, min_overlap=min_overlap,
                                                  thresh=0, compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                all_thresholds[m, l, k, :len(thresholds)] = thresholds
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(parted_overlaps[j], pr, total_gt_num[idx:idx + num_part],
                                             total_dt_num[idx:idx + num_part], total_dc_num[idx:idx + num_part],
                                             gt_datas_part, dt_datas_part, dc_datas_part, ignored_gts_part,
                                             ignored_dets_part, metric, min_overlap=min_overlap, thresholds=thresholds,
                                             compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)

    ret_dict = {'precision': precision, 'orientation': aos, 'thresholds': all_thresholds, 'min_overlaps': min_overlaps}
    return ret_dict


def do_eval(gt_annos, dt_annos, current_classes, min_overlaps, compute_aos=False, difficulties=(0, 1, 2),
            z_axis=1, z_center=1.0):
    types = ['bbox', 'bev', '3d']
    metrics = {}
    for i in range(3):
        metrics[types[i]] = eval_class(gt_annos, dt_annos, current_classes, difficulties, i, min_overlaps, compute_aos,
                                       z_axis=z_axis, z_center=z_center)
    return metrics


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def get_official_eval_result(gt_annos, dt_annos, current_classes, difficulties=(0, 1, 2), z_axis=1, z_center=1.0):
    """
    :param gt_annos: must contains following keys: [bbox, location, dimensions, rotation_y, score]
    :param dt_annos: must contains following keys: [bbox, location, dimensions, rotation_y, score]
    :param current_classes:
    :param difficulties:
    :param z_axis:
    :param z_center:
    :return:
    """
    min_overlaps = np.array([[[0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                              [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                              [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7]]])
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'car',
        6: 'tractor',
        7: 'trailer',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for cur_cls in current_classes:
        if isinstance(cur_cls, str):
            current_classes_int.append(name_to_class[cur_cls])
        else:
            current_classes_int.append(cur_cls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    metrics = do_eval(gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, difficulties,
                      z_axis=z_axis, z_center=z_center)
    results_str = ''
    results = dict()
    for j, cur_cls in enumerate(current_classes):
        cur_cls_name = class_to_name[cur_cls]
        # mAP threshold array: [num_min_overlap, metric, class]
        # mAP result: [num_class, num_diff, num_min_overlap]
        map_bbox = get_map(metrics['bbox']['precision'][j, :, 0])
        map_bbox_str = ', '.join(f'{v:.2f}' for v in map_bbox)
        map_bev = get_map(metrics['bev']['precision'][j, :, 0])
        map_bev_str = ', '.join(f'{v:.2f}' for v in map_bev)
        map_3d = get_map(metrics['3d']['precision'][j, :, 0])
        map_3d_str = ', '.join(f'{v:.2f}' for v in map_3d)
        results_str += print_str((f'{cur_cls_name}'
                                  ' AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:'.format(*min_overlaps[0, :, j])))
        results_str += print_str(f'bbox AP:{map_bbox_str}')
        results_str += print_str(f'bev  AP:{map_bev_str}')
        results_str += print_str(f'3d   AP:{map_3d_str}')
        if compute_aos:
            map_aos = get_map(metrics['bbox']['orientation'][j, :, 0])
            map_aos = ', '.join(f'{v:.2f}' for v in map_aos)
            results_str += print_str(f'aos  AP:{map_aos}')
        results[cur_cls_name] = {'bbox': map_bbox, 'bev': map_bev, '3d': map_3d}

    return metrics, results, results_str
