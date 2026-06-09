"""Jeu de survie aligné sur l'embedding CT-FM : joint l'embedding figé (512-D, déjà poolé
par l'extracteur) de chaque patient à sa cible de survie (temps, événement) via le helper
partagé `src.survival_targets.survival_xy`.
"""
import numpy as np
import pandas as pd
import torch

from src.survival_targets import survival_xy


def load_foundation_survival(config) -> tuple:
    """Charge les embeddings CT-FM figés des deux splits et les joint à la cible de survie."""
    # weights_only=False : features produites par notre propre CtFmExtractor.
    train = torch.load(config.foundation_train_features_path, map_location="cpu", weights_only=False)
    val = torch.load(config.foundation_val_features_path, map_location="cpu", weights_only=False)
    patients = pd.read_csv(config.csv_path)
    train_xy = survival_xy(np.asarray(train["embedding"], dtype=np.float32), train["case_id"], patients)
    val_xy = survival_xy(np.asarray(val["embedding"], dtype=np.float32), val["case_id"], patients)
    print(f"CT-FM survival: {len(train_xy[1])} train / {len(val_xy[1])} val patients with RFS "
          f"(dim {train_xy[0].shape[1]})")
    return train_xy, val_xy
