"""Jeu de données d'embeddings TEP/CT figés pour les stades T/N.

Lit les features poolées de `tables/bottleneck.csv` (via `src.features`) et les aligne par
case_id avec les cibles de stade T et N lues dans le CSV clinique.
"""
import numpy as np
import pandas as pd

from src.features import load_features_by_split

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


class EmbeddingDataset:
    """Embedding TEP/CT figé d'un split, aligné par case_id avec ses cibles de stade T/N.
    Alimente les RandomForest de `tn.train` (la survie passe par les données cliniques)."""

    def __init__(self, case_ids: list, embeddings: np.ndarray, patients: pd.DataFrame):
        rows = patients.set_index("PatientID")

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
    """Croise `tables/bottleneck.csv` avec le split du CSV clinique et joint les features à
    leurs cibles de stade T/N."""
    patients = pd.read_csv(config.csv_path)
    features = load_features_by_split(config, patients)
    train = EmbeddingDataset(*features["train"], patients)
    test = EmbeddingDataset(*features["test"], patients)
    print(f"loaded {len(train)} train and {len(test)} test embeddings (dim {train.X.shape[1]})")
    return train, test
