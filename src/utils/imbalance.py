import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def compute_weights_from_labels(labels):
    # labels: list/np.array of ints (0/1)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weight_dict = {int(c): float(w) for c,w in zip(classes, weights)}
    return class_weight_dict
