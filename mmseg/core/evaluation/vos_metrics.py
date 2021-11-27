import torch
from torch import Tensor

import numpy as np
from math import floor
from collections import OrderedDict

from .metrics import eval_metrics, total_intersect_and_union, f_score
from skimage.morphology import binary_dilation, disk

import warnings
from joblib import delayed, Parallel


def vos_eval_metrics(results,
                     gt_seg_maps,
                     num_classes,
                     ignore_index,
                     metrics=['mIoU'],
                     nan_to_num=None,
                     label_map=dict(),
                     reduce_zero_label=False,
                     beta=1,
                     ):
    my_metrics = ['VOS']
    original_allowed_metrics = ['mIoU', 'mDice', 'mFscore', ]
    allowed_metrics = my_metrics + original_allowed_metrics

    if isinstance(metrics, str):
        metrics = [metrics]

    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    ret_metrics = OrderedDict()
    for metric in metrics:
        if set([metric]).issubset(set(original_allowed_metrics)):
            ret_metrics.update(eval_metrics(results,
                                            gt_seg_maps,
                                            num_classes,
                                            ignore_index,
                                            metric,
                                            nan_to_num,
                                            label_map,
                                            reduce_zero_label,
                                            beta))
        elif set([metric]).issubset(set(my_metrics)):
            # total_area_intersect, total_area_union, total_area_pred_label, \
            # total_area_label = total_intersect_and_union(
            #     results, gt_seg_maps, num_classes, ignore_index, label_map,
            #     reduce_zero_label)
            #
            # all_acc = total_area_intersect.sum() / total_area_label.sum()
            # ret_metrics.update(OrderedDict({'aAcc': all_acc}))
            num_imgs = len(results)
            assert len(gt_seg_maps) == num_imgs
            frame_score_list = Parallel(n_jobs=40)(
                delayed(_read_and_eval_file)(
                    frame_mask=gt_seg_maps[i], frame_pred=results[i]
                )
                for i in range(num_imgs)
            )

            # frame_score_list = [_read_and_eval_file(
            #     frame_mask=gt_seg_maps[i], frame_pred=results[i])
            #     for i in range(num_imgs)
            # ]
            frame_score_array = np.asarray(frame_score_list)
            M, O = zip(
                *[
                    get_mean_recall_decay_for_video(frame_score_array[:, i])
                    for i in range(frame_score_array.shape[1])
                ]
            )
            if metric == 'VOS':
                ret_metrics['JM'] = M[0]
                ret_metrics['JO'] = O[0]
                ret_metrics['FM'] = M[1]
                ret_metrics['FO'] = O[1]

    for metric, value in ret_metrics.copy().items():
        if isinstance(value, Tensor):
            ret_metrics.update({metric: value.numpy()})

    if nan_to_num is not None:
        ret_metrics.update(OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        }))

    return ret_metrics


def _read_and_eval_file(frame_mask, frame_pred):
    binary_frame_mask = (frame_mask > 0).astype(np.float32)
    binary_frame_pred = (frame_pred > 0).astype(np.float32)
    J_score = db_eval_iou(
        annotation=binary_frame_mask, segmentation=binary_frame_pred
    )
    F_score = db_eval_boundary(
        foreground_mask=binary_frame_pred, gt_mask=binary_frame_mask
    )
    return J_score, F_score


def get_mean_recall_decay_for_video(per_frame_values):
    """Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation
    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values[1:-1] > 0.5)

    # # Compute decay as implemented in Matlab
    # per_frame_values = per_frame_values[1:-1]  # Remove first frame
    #
    # N_bins = 4
    # ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    # ids = ids.astype(np.uint8)
    #
    # D_bins = [per_frame_values[ids[i]: ids[i + 1] + 1] for i in range(0, 4)]
    #
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    #     D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O


# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
# -----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------
def db_eval_boundary(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = (
        bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))
    )

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width      :    Width of desired bmap  <= seg.shape[1]
        height  :    Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):    Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
            width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (
        w,
        h,
        width,
        height,
    )

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + floor((y - 1) + height / h)
                    i = 1 + floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def db_eval_iou(annotation, segmentation):
    """Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
    """

    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / np.sum(
            (annotation | segmentation), dtype=np.float32
        )
