"""Lecture des features poolées du bottleneck nnU-Net (`tables/bottleneck.csv`).

Le CSV — produit hors-ligne sur le cluster par `seg.extract` puis déposé dans `tables/` —
contient une ligne par patient (`PatientID` + une colonne `feat_i` par dimension), tous
splits confondus. Le découpage train/test est celui de la colonne `split` du CSV clinique.

Ce module (pandas seul, sans torch ni nnU-Net) est l'UNIQUE porte d'entrée des features pour
`tn` et `nnunet_survival` : plus aucune extraction ailleurs dans le dépôt.
"""
import numpy as np
import pandas as pd

FEATURE_PREFIX = "feat_"


def load_features_by_split(config, patients: pd.DataFrame) -> dict:
    """Croise `tables/bottleneck.csv` avec la colonne `split` du CSV clinique `patients`.
    Renvoie {"train": (case_ids, X), "test": (case_ids, X)} — X étant la matrice float32 des
    colonnes `feat_i`, alignée sur `case_ids`."""
    df = pd.read_csv(config.bottleneck_csv_path)
    df["PatientID"] = df["PatientID"].astype(str)
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    split_of = dict(zip(patients["PatientID"].astype(str), patients["split"].astype(str)))

    result = {}
    for split in ("train", "test"):
        sub = df[df["PatientID"].map(split_of) == split]
        result[split] = (sub["PatientID"].tolist(), sub[feature_cols].to_numpy(dtype=np.float32))
    return result
