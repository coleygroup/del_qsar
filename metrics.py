from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

def get_roc_auc(true_labels, preds):
    return roc_auc_score(true_labels, preds)

def get_pr_auc(true_labels, preds):
    precision, recall, _ = precision_recall_curve(true_labels, preds)
    return auc(recall, precision)
