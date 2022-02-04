from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy.stats import spearmanr
import torch
import numpy as np

def get_roc_auc(true_labels, preds):
    return roc_auc_score(true_labels, preds)

def get_pr_auc(true_labels, preds):
    precision, recall, _ = precision_recall_curve(true_labels, preds)
    return auc(recall, precision)

def get_spearman_r(R, k1, k2, n1, n2):
    a = np.power(0, 2) / 4 - (k2 + 3/8)
    b = 2 * torch.sqrt(k1 + 3/8) * torch.sqrt(k2 + 3/8)
    c = np.power(0, 2) / 4 - (k1 + 3/8)
    x = (-b) / (2*a)
    R_target = torch.pow(x, 2) * n2/n1
    correlation, pvalue = spearmanr(R, R_target)
    return correlation
