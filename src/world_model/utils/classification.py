import numpy as np


def binary_metrics(pred, label):
    """Confusion counts and the summary scores, for boolean predictions against boolean labels.

    F1 alone is unsafe on a skewed set: on memtest4_eval (positive rate 0.359) "always ball"
    scores F1 0.528, and a near-constant predictor once scored 0.544 while its accuracy fell to
    0.410 -- below every baseline. balanced_accuracy averages the two class recalls, so a
    constant predictor scores 0.5 whatever the prior; mcc is 0 for any constant predictor and
    is the one number that collapses the whole confusion matrix honestly.
    """
    pred, label = np.asarray(pred, dtype=bool), np.asarray(label, dtype=bool)
    tp = int((pred & label).sum())
    fp = int((pred & ~label).sum())
    tn = int((~pred & ~label).sum())
    fn = int((~pred & label).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    # every factor zero <=> one predicted or one true class is empty, i.e. the matrix is
    # degenerate and no correlation is defined; 0.0 is the conventional reading
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / np.sqrt(denominator) if denominator > 0 else 0.0
    return {"n": int(pred.size), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "accuracy": (tp + tn) / pred.size if pred.size else 0.0,
            "specificity": specificity, "balanced_accuracy": 0.5 * (recall + specificity), "mcc": float(mcc)}
