"""Jeu de données d'embeddings TEP/CT figés pour les stades T/N.

Charge les features extraites par `tn.extractor` (bottleneck de l'encodeur nnU-Net), les
réduit en un vecteur par patient (moyenne ⊕ max global) et les aligne par case_id avec les
cibles de stade T et N lues dans le CSV.
"""
import numpy as np
import pandas as pd
import torch

T_STAGES = ["T1", "T2", "T3", "T4"]
N_STAGES = ["N0", "N1", "N2", "N3"]
UNKNOWN_STAGE = -1


def _stage_index(value, stages: list) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return UNKNOWN_STAGE
    label = str(value).strip().upper()
    if label.startswith("N2"):
        label = "N2"
    return stages.index(label) if label in stages else UNKNOWN_STAGE


def pool_embedding(bottleneck: torch.Tensor) -> np.ndarray:
    """Réduit le bottleneck spatial (N, C, D, H, W) en un vecteur par patient.
    On concatène moyenne globale (contexte) et maximum global (pic d'activation,
    proche d'un SUVmax) : l'embedding TEP/CT figé servi tel quel aux forêts."""
    mean = bottleneck.mean(dim=(2, 3, 4))
    peak = bottleneck.amax(dim=(2, 3, 4))
    return torch.cat([mean, peak], dim=1).numpy()


class EmbeddingDataset:
    """Embedding TEP/CT figé d'un split, aligné par case_id avec ses cibles de stade T/N.
    Alimente les RandomForest de `tn.train` (la survie passe par les données cliniques)."""

    def __init__(self, features: dict, patients: pd.DataFrame):
        rows = patients.set_index("PatientID")
        case_ids = list(features["case_id"])
        embeddings = pool_embedding(features["bottleneck"])

        keep, t, n = [], [], []
        for i, case_id in enumerate(case_ids):
            if case_id not in rows.index:
                continue
            keep.append(i)
            row = rows.loc[case_id]
            t.append(_stage_index(row.get("T-stage"), T_STAGES))
            n.append(_stage_index(row.get("N-stage"), N_STAGES))

        self.case_ids = [case_ids[i] for i in keep]
        self.X = embeddings[keep]
        self.t_label = np.array(t, dtype=np.int64)
        self.n_label = np.array(n, dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def labelled(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        """Sous-ensemble (X, y) du champ de stade pour lequel le label est connu (>= 0)."""
        y = getattr(self, field)
        mask = y >= 0
        return self.X[mask], y[mask]


def load_embeddings(config) -> tuple[EmbeddingDataset, EmbeddingDataset]:
    """Charge les embeddings figés des deux splits et les joint à leurs cibles."""
    # weights_only=False : features produites par notre propre NNUNetBottleneckExtractor.
    train_features = torch.load(config.train_features_path, map_location="cpu", weights_only=False)
    val_features = torch.load(config.val_features_path, map_location="cpu", weights_only=False)
    patients = pd.read_csv(config.csv_path)
    train = EmbeddingDataset(train_features, patients)
    val = EmbeddingDataset(val_features, patients)
    print(f"loaded {len(train)} train and {len(val)} val embeddings (dim {train.X.shape[1]})")
    return train, val
