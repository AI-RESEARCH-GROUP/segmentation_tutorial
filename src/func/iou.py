import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def iou(label_trues, label_preds, n_class):
    # label_preds,label_trues = output #此处的label_pred的shape是（batch_size,nclass,...）
    # label_preds = label_preds.max(dim=1)[1].cpu().numpy() #若label的shape为(b,n_class,h,w),则label.max(dim=1)返回的是第一维中每一列的最大值及其索引，values和indices的shape为（b,h,w）

    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    with np.errstate(divide='ignore', invalid='ignore'):  # 忽略0为分母时的错误
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)
    return mean_iou
