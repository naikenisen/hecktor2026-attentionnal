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


def bootstrap_c_index(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """IC 95% du c-index par bootstrap (percentile method). Retourne (ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(times))
    samples = [
        c_index(risk_scores[s], times[s], events[s])
        for s in (rng.choice(idx, size=len(idx), replace=True) for _ in range(n))
    ]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_c_index_diff(
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """IC 95% de (c-index_a − c-index_b) par bootstrap apparié. Retourne (ci_low, ci_high).
    Si 0 est hors de l'intervalle, les deux modèles sont statistiquement distinguables."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(times))
    diffs = [
        c_index(risk_a[s], times[s], events[s]) - c_index(risk_b[s], times[s], events[s])
        for s in (rng.choice(idx, size=len(idx), replace=True) for _ in range(n))
    ]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
