# src/utils/metrics.py
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

def compute_metrics(pred):
    logits = pred.predictions
    if isinstance(logits, tuple):  # sometimes HF returns (logits, hidden)
        logits = logits[0]
    probs = softmax(logits, axis=1)[:, 1]
    y_pred = np.argmax(logits, axis=1)
    y_true = pred.label_ids
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, probs)
    except:
        auc = 0.0
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f, "roc_auc": auc}

def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)
