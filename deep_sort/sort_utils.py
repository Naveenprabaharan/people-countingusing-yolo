# FILE: deep_sort/sort_utils.py

import numpy as np


def iou(bb_test, bb_gt):
    """Compute IOU between two tlbr boxes"""
    bb_test = np.asarray(bb_test, dtype=np.float32)
    bb_gt = np.asarray(bb_gt, dtype=np.float32)
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union