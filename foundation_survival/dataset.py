"""Jeu de survie aligné sur l'embedding CT-FM : joint l'embedding figé de chaque patient à
sa cible de survie (temps, événement), en ne gardant que les patients dont le RFS est
renseigné (> 0).
"""
import numpy as np
import pandas as pd
import torch

import config


def _survival_xy(features: dict, patients: pd.DataFrame) -> tuple:
    """Aligne l'embedding CT-FM de chaque case_id avec sa cible (RFS, événement)."""
    rows = patients.set_index("PatientID")
    embeddings = np.asarray(features["embedding"], dtype=np.float32)
    X, time, event = [], [], []
    for i, case_id in enumerate(features["case_id"]):
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


def load_foundation_survival(config) -> tuple:
    """Charge les embeddings CT-FM figés des deux splits et les joint à la cible de survie."""
    # weights_only=False : features produites par notre propre CtFmExtractor.
    train = torch.load(config.foundation_train_features_path, map_location="cpu", weights_only=False)
    val = torch.load(config.foundation_val_features_path, map_location="cpu", weights_only=False)
    patients = pd.read_csv(config.csv_path)
    train_xy = _survival_xy(train, patients)
    val_xy = _survival_xy(val, patients)
    print(f"CT-FM survival: {len(train_xy[1])} train / {len(val_xy[1])} val patients with RFS "
          f"(dim {train_xy[0].shape[1]})")
    return train_xy, val_xy
