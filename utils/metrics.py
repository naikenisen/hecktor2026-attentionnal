import numpy as np
from sklearn.metrics import balanced_accuracy_score
from lifelines.utils import concordance_index

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true >= 0
    if mask.sum() == 0:
        return 0.0
    return float(balanced_accuracy_score(y_true[mask], y_pred[mask]))

def c_index(risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    finite = np.isfinite(risk_scores) & np.isfinite(times) & np.isfinite(events)
    risk_scores, times, events = risk_scores[finite], times[finite], events[finite]
    if len(times) == 0 or events.sum() == 0:
        return 0.5
    return float(concordance_index(times, -risk_scores, events))
