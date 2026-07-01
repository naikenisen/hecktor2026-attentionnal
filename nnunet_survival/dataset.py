"""Jeu de survie aligné sur le bottleneck nnU-Net.

Lit les features poolées de `tables/bottleneck.csv` (via `src.features`, le même CSV que
`tn`) et les aligne avec la cible de survie via le helper partagé
`src.survival_targets.survival_xy`.
"""
import pandas as pd

from src.features import load_features_by_split
from src.survival_targets import survival_xy


def load_nnunet_survival(config) -> tuple:
    """Croise `tables/bottleneck.csv` avec le split du CSV clinique et joint les features à
    la cible de survie (RFS, événement)."""
    patients = pd.read_csv(config.csv_path)
    features = load_features_by_split(config, patients)
    train_ids, train_X = features["train"]
    test_ids, test_X = features["test"]
    train_xy = survival_xy(train_X, train_ids, patients)
    test_xy = survival_xy(test_X, test_ids, patients)
    print(f"nnU-Net survival: {len(train_xy[1])} train / {len(test_xy[1])} test patients with RFS "
          f"(dim {train_xy[0].shape[1]})")
    return train_xy, test_xy
