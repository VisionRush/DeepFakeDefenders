import math
import torch
import numpy as np
import sklearn
import sklearn.metrics


class MultilabelClassificationMetric(object):
    def __init__(self):
        super(MultilabelClassificationMetric, self).__init__()
        self.pred_scores_ = torch.FloatTensor()    # .FloatStorage()
        self.grth_labels_ = torch.LongTensor()     # .LongStorage()

    # Func:
    #   Reset calculation.
    def reset(self):
        self.pred_scores_ = torch.FloatTensor(torch.FloatStorage())
        self.grth_labels_ = torch.LongTensor(torch.LongStorage())

    # Func:
    #   Add prediction and groundtruth that will be used to calculate average precision.
    # Input:
    #   pred_scores  : predicted scores,   size: [batch_size, label_dim], format: [s0, s1, ..., s19]
    #   grth_labels  : groundtruth labels, size: [batch_size, label_dim], format: [c0, c1, ..., c19]
    def add(self, pred_scores, grth_labels):
        if not torch.is_tensor(pred_scores):
            pred_scores = torch.from_numpy(pred_scores)
        if not torch.is_tensor(grth_labels):
            grth_labels = torch.from_numpy(grth_labels)

        # check
        assert pred_scores.dim() == 2, 'wrong pred_scores size (should be 2D with format: [batch_size, label_dim(one column per class)])'
        assert grth_labels.dim() == 2, 'wrong grth_labels size (should be 2D with format: [batch_size, label_dim(one column per class)])'

        # check storage is sufficient
        if self.pred_scores_.storage().size() < self.pred_scores_.numel() + pred_scores.numel():
            new_size = math.ceil(self.pred_scores_.storage().size() * 1.5)
            self.pred_scores_.storage().resize_(int(new_size + pred_scores.numel()))
            self.grth_labels_.storage().resize_(int(new_size + pred_scores.numel()))

        # store outputs and targets
        offset = self.pred_scores_.size(0) if self.pred_scores_.dim() > 0 else 0
        self.pred_scores_.resize_(offset + pred_scores.size(0), pred_scores.size(1))
        self.grth_labels_.resize_(offset + grth_labels.size(0), grth_labels.size(1))
        self.pred_scores_.narrow(0, offset, pred_scores.size(0)).copy_(pred_scores)
        self.grth_labels_.narrow(0, offset, grth_labels.size(0)).copy_(grth_labels)

    # Func:
    #   Compute average precision.
    def calc_avg_precision(self):
        # check
        if self.pred_scores_.numel() == 0: return 0
        # calc by class
        aps = torch.zeros(self.pred_scores_.size(1))
        for cls_idx in range(self.pred_scores_.size(1)):
            # get pred scores & grth labels at class cls_idx
            cls_pred_scores = self.pred_scores_[:, cls_idx]    # predictions for all images at class cls_idx, format: [img_num]
            cls_grth_labels = self.grth_labels_[:, cls_idx]    # truthvalues for all iamges at class cls_idx, format: [img_num]
            # sort by socre
            _, img_indices = torch.sort(cls_pred_scores, dim=0, descending=True)
            # calc ap
            TP, TPFP = 0., 0.
            for img_idx in img_indices:
                label = cls_grth_labels[img_idx]
                # accumulate
                TPFP += 1
                if label == 1:
                    TP += 1
                    aps[cls_idx] += TP / TPFP
            aps[cls_idx] /= (TP + 1e-5)
        # return
        return aps

    # Func:
    #   Compute average precision.
    def calc_avg_precision2(self):
        self.pred_scores_ = self.pred_scores_.cpu().numpy().astype('float32')
        self.grth_labels_ = self.grth_labels_.cpu().numpy().astype('float32')
        # check
        if self.pred_scores_.size == 0: return 0
        # calc by class
        aps = np.zeros(self.pred_scores_.shape[1])
        for cls_idx in range(self.pred_scores_.shape[1]):
            # get pred scores & grth labels at class cls_idx
            cls_pred_scores = self.pred_scores_[:, cls_idx]
            cls_grth_labels = self.grth_labels_[:, cls_idx]
            # compute ap for a object category
            aps[cls_idx] = sklearn.metrics.average_precision_score(cls_grth_labels, cls_pred_scores)
        aps[np.isnan(aps)] = 0
        aps = np.around(aps, decimals=4)
        return aps


class MultiClassificationMetric(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        super(MultiClassificationMetric, self).__init__()
        self.reset()
        self.val = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean, self.std = self.sum, np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = math.sqrt(self.m_s / (self.n - 1.0))

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


def simple_accuracy(output, target):
    """计算预测正确的准确率"""
    with torch.no_grad():
        _, preds = torch.max(output, 1)

        correct = preds.eq(target).float()
        accuracy = correct.sum() / len(target)
        return accuracy