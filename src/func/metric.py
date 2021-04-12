import numpy as np


def mean_iou(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


def gen_fusion_matrix(pred_image, gt_image, num_class):
    confusion_matrix = np.zeros((num_class,) * 2)
    for pred, gt in zip(pred_image, gt_image):
        mask = (gt >= 0) & (gt < num_class)
        label = num_class * gt[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=num_class ** 2)
        confusion_matrix += count.reshape(num_class, num_class)
    return confusion_matrix

