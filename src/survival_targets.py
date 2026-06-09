"""Alignement embedding ↔ cible de survie (RFS, événement).

Partagé par les têtes de survie qui partent d'une matrice d'embeddings figés :
`foundation_survival` (CT-FM) et `nnunet_survival` (bottleneck nnU-Net). La survie clinique
n'utilise pas ce helper (elle encode des variables tabulaires, cf. clinical_survival).
"""
import numpy as np
import pandas as pd


def survival_xy(embeddings: np.ndarray, case_ids: list, patients: pd.DataFrame) -> tuple:
    """Aligne une matrice d'embeddings (N, D) avec la cible (temps, événement) de survie,
    en ne gardant que les patients dont le RFS est renseigné (> 0)."""
    rows = patients.set_index("PatientID")
    X, time, event = [], [], []
    for i, case_id in enumerate(case_ids):
        if case_id not in rows.index:
            continue
        row = rows.loc[case_id]
        relapse_free_survival = row.get("RFS", np.nan)
        if pd.isna(relapse_free_survival) or float(relapse_free_survival) <= 0:
            continue
        X.append(embeddings[i])
        time.append(float(relapse_free_survival))
        relapse = row.get("Relapse", np.nan)
        event.append(int(relapse) if not pd.isna(relapse) else 0)
    return np.stack(X), np.asarray(time, dtype=np.float64), np.asarray(event, dtype=bool)
