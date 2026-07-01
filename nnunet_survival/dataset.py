"""Jeu de survie aligné sur le bottleneck nnU-Net.

Réutilise les features extraites par `src.nnunet_embedding` (les mêmes que `tn`), les réduit
en un vecteur par patient (`pool_embedding`) et les aligne avec la cible de survie via le
helper partagé `src.survival_targets.survival_xy`.
"""
import pandas as pd
import torch

from src.nnunet_embedding import pool_embedding
from src.survival_targets import survival_xy


def load_nnunet_survival(config) -> tuple:
    """Charge les embeddings du bottleneck nnU-Net des deux splits et les joint à la cible
    de survie (RFS, événement)."""
    # weights_only=False : features produites par notre propre NNUNetBottleneckExtractor.
    train = torch.load(config.train_features_path, map_location="cpu", weights_only=False)
    test = torch.load(config.test_features_path, map_location="cpu", weights_only=False)
    patients = pd.read_csv(config.csv_path)
    train_xy = survival_xy(pool_embedding(train["bottleneck"]), train["case_id"], patients)
    test_xy = survival_xy(pool_embedding(test["bottleneck"]), test["case_id"], patients)
    print(f"nnU-Net survival: {len(train_xy[1])} train / {len(test_xy[1])} test patients with RFS "
          f"(dim {train_xy[0].shape[1]})")
    return train_xy, test_xy
