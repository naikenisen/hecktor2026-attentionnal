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
    """Embedding TEP/CT figé d'un split, aligné par case_id avec ses cibles tabulaires
    (stades T/N, temps et événement de survie). Aucune donnée clinique fusionnée."""

    def __init__(self, features: dict, patients: pd.DataFrame):
        rows = patients.set_index("PatientID")
        case_ids = list(features["case_id"])
        embeddings = pool_embedding(features["bottleneck"])

        keep, t, n, time, event = [], [], [], [], []
        for i, case_id in enumerate(case_ids):
            if case_id not in rows.index:
                continue
            row = rows.loc[case_id]
            keep.append(i)
            t.append(_stage_index(row.get("T-stage"), T_STAGES))
            n.append(_stage_index(row.get("N-stage"), N_STAGES))
            relapse_free_survival = row.get("RFS", np.nan)
            time.append(float(relapse_free_survival) if not pd.isna(relapse_free_survival) else 0.0)
            relapse = row.get("Relapse", np.nan)
            event.append(int(relapse) if not pd.isna(relapse) else 0)

        self.case_ids = [case_ids[i] for i in keep]
        self.X = embeddings[keep]
        self.t_label = np.array(t, dtype=np.int64)
        self.n_label = np.array(n, dtype=np.int64)
        self.time = np.array(time, dtype=np.float64)
        self.event = np.array(event, dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def labelled(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        """Sous-ensemble (X, y) du champ de stade pour lequel le label est connu (>= 0)."""
        y = getattr(self, field)
        mask = y >= 0
        return self.X[mask], y[mask]

    def survival(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sous-ensemble (X, time, event) des patients dont le RFS est renseigné (> 0)."""
        mask = self.time > 0
        return self.X[mask], self.time[mask], self.event[mask].astype(bool)


def load_embeddings(config) -> tuple[EmbeddingDataset, EmbeddingDataset]:
    """Charge les embeddings figés des deux splits et les joint à leurs cibles."""
    # weights_only=False : features produites par notre propre BottleneckExtractor.
    train_features = torch.load(config.train_features_path, map_location="cpu", weights_only=False)
    val_features = torch.load(config.val_features_path, map_location="cpu", weights_only=False)
    patients = pd.read_csv(config.csv_path)
    train = EmbeddingDataset(train_features, patients)
    val = EmbeddingDataset(val_features, patients)
    print(f"loaded {len(train)} train and {len(val)} val embeddings (dim {train.X.shape[1]})")
    return train, val
