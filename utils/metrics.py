import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from lifelines.utils import concordance_index

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    mask = y_true >= 0
    if mask.sum() == 0:
        return 0.0
    return float(balanced_accuracy_score(y_true[mask], y_pred[mask]))

def discrete_risk_from_surv_logits(surv_logits: torch.Tensor) -> torch.Tensor:

    p = torch.softmax(surv_logits, dim=-1)

    T = surv_logits.size(-1)

    bins = torch.arange(T, device=surv_logits.device, dtype=p.dtype) + 1.0

    expected_bin = (p * bins).sum(dim=-1)

    return -expected_bin

def c_index(risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:

    finite = np.isfinite(risk_scores) & np.isfinite(times) & np.isfinite(events)
    risk_scores, times, events = risk_scores[finite], times[finite], events[finite]
    if len(times) == 0 or events.sum() == 0:
        return 0.5

    return float(concordance_index(times, -risk_scores, events))
