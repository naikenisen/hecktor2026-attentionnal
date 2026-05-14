import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from lifelines.utils import concordance_index

# Calcule la Balanced Accuracy en ignorant les labels −1 (inconnus)
def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Masque pour exclure les patients dont le stage est inconnu (label = −1)
    mask = y_true >= 0
    if mask.sum() == 0:
        return 0.0
    return float(balanced_accuracy_score(y_true[mask], y_pred[mask]))


# Convertit les logits de survie (B, T) en score de risque scalaire (B,) par espérance inversée
def discrete_risk_from_surv_logits(surv_logits: torch.Tensor) -> torch.Tensor:
    # Distribution de probabilité softmax sur les T bins
    p = torch.softmax(surv_logits, dim=-1)
    # Nombre de bins temporels
    T = surv_logits.size(-1)
    # Indices des bins (1..T) utilisés comme valeurs temporelles relatives
    bins = torch.arange(T, device=surv_logits.device, dtype=p.dtype) + 1.0
    # Espérance du bin : élevée si l'événement est attendu tardivement (faible risque)
    expected_bin = (p * bins).sum(dim=-1)
    # Signe inversé : score élevé ↔ risque élevé ↔ événement attendu tôt
    return -expected_bin


# Calcule le C-index de concordance (Harrell) entre scores de risque et temps de survie
def c_index(risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    if events.sum() == 0:
        return 0.5
    # lifelines attend que scores élevés ↔ longue durée → on passe −risk
    return float(concordance_index(times, -risk_scores, events))
