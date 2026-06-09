import os
import random
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


def split_case_ids(config) -> tuple[list, list]:
    """Split train/val déterministe des patients présents dans data_root (CPU seul,
    sans MONAI). Source unique du découpage, partagée par la pipeline image (extraction
    des embeddings) et la survie clinique : garantit des splits strictement identiques."""
    case_ids = sorted(
        d for d in os.listdir(config.data_root)
        if os.path.isdir(os.path.join(config.data_root, d))
    )
    random.seed(config.seed)
    random.shuffle(case_ids)
    n_val = int(len(case_ids) * config.val_split)
    return case_ids[n_val:], case_ids[:n_val]


def pool_embedding(bottleneck: torch.Tensor) -> np.ndarray:
    """Réduit le bottleneck spatial (N, C, D, H, W) en un vecteur par patient.
    On concatène moyenne globale (contexte) et maximum global (pic d'activation,
    proche d'un SUVmax) : l'embedding TEP/CT figé servi tel quel aux forêts."""
    mean = bottleneck.mean(dim=(2, 3, 4))
    peak = bottleneck.amax(dim=(2, 3, 4))
    return torch.cat([mean, peak], dim=1).numpy()


class EmbeddingDataset:
    """Embedding TEP/CT figé d'un split, aligné par case_id avec ses cibles de stade T/N.
    Alimente les RandomForest de train_tn.py (la survie passe par les données cliniques)."""

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


# ─────────────────────────── Survie sur données cliniques ───────────────────────────
# La survie n'utilise AUCUNE information image : seulement les variables tabulaires du
# CSV. T-stage / N-stage sont exclus (ce sont les cibles de train_tn et ils ne sont pas
# disponibles à l'inférence).

AGE_COLUMN = "Age"
CATEGORICAL_COLUMNS = [
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "HPV Status",
    "Treatment",
]
UNKNOWN_CATEGORY = "Inconnu"


class ClinicalEncoder:
    """Apprend sur le train les statistiques d'âge et les catégories observées, puis
    transforme chaque patient en vecteur numérique (âge standardisé + one-hot).
    Les valeurs manquantes tombent dans une catégorie « Inconnu » dédiée (âge imputé
    par la médiane du train)."""

    def __init__(self):
        self.age_mean = 0.0
        self.age_std = 1.0
        self.age_median = 0.0
        self.category_indices = {}

    def fit(self, patients: pd.DataFrame):
        ages = patients[AGE_COLUMN].dropna().to_numpy(dtype=float)
        if len(ages):
            self.age_median = float(np.median(ages))
            self.age_mean = float(ages.mean())
            self.age_std = float(ages.std()) if ages.std() > 1e-6 else 1.0
        for column in CATEGORICAL_COLUMNS:
            values = patients[column].dropna().astype(str).unique().tolist()
            if UNKNOWN_CATEGORY not in values:
                values.append(UNKNOWN_CATEGORY)
            self.category_indices[column] = {value: i for i, value in enumerate(sorted(values))}
        return self

    @property
    def output_dim(self) -> int:
        return 1 + sum(len(indices) for indices in self.category_indices.values())

    def encode_row(self, row: pd.Series) -> np.ndarray:
        age = row.get(AGE_COLUMN, np.nan)
        age = self.age_median if pd.isna(age) else float(age)
        features = [(age - self.age_mean) / self.age_std]
        for column, indices in self.category_indices.items():
            value = row.get(column, np.nan)
            value = UNKNOWN_CATEGORY if pd.isna(value) else str(value)
            position = indices.get(value, indices[UNKNOWN_CATEGORY])
            one_hot = [0.0] * len(indices)
            one_hot[position] = 1.0
            features.extend(one_hot)
        return np.array(features, dtype=np.float32)


class ClinicalSurvivalDataset:
    """Variables cliniques encodées d'un split, alignées par case_id avec (temps, événement)
    de survie sans rechute. Aucune donnée image."""

    def __init__(self, case_ids: list, patients: pd.DataFrame, encoder: ClinicalEncoder):
        rows = patients.set_index("PatientID")
        features, time, event, kept = [], [], [], []
        for case_id in case_ids:
            if case_id not in rows.index:
                continue
            row = rows.loc[case_id]
            features.append(encoder.encode_row(row))
            relapse_free_survival = row.get("RFS", np.nan)
            time.append(float(relapse_free_survival) if not pd.isna(relapse_free_survival) else 0.0)
            relapse = row.get("Relapse", np.nan)
            event.append(int(relapse) if not pd.isna(relapse) else 0)
            kept.append(case_id)
        self.case_ids = kept
        self.X = np.stack(features) if features else np.empty((0, encoder.output_dim), np.float32)
        self.time = np.array(time, dtype=np.float64)
        self.event = np.array(event, dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def survival(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sous-ensemble (X, time, event) des patients dont le RFS est renseigné (> 0)."""
        mask = self.time > 0
        return self.X[mask], self.time[mask], self.event[mask].astype(bool)


def load_clinical_survival(config) -> tuple[ClinicalSurvivalDataset, ClinicalSurvivalDataset]:
    """Construit les jeux de survie cliniques train/val. Le split (split_case_ids) est
    identique à celui de la pipeline image, pour rester cohérent avec train_tn, mais
    aucune image — ni MONAI — n'est chargée ici. L'encodeur est ajusté sur le train."""
    train_ids, val_ids = split_case_ids(config)
    patients = pd.read_csv(config.csv_path)
    train_patients = patients[patients["PatientID"].isin(train_ids)]
    encoder = ClinicalEncoder().fit(train_patients)
    train = ClinicalSurvivalDataset(train_ids, patients, encoder)
    val = ClinicalSurvivalDataset(val_ids, patients, encoder)
    print(f"loaded clinical features (dim {encoder.output_dim}): "
          f"{len(train)} train / {len(val)} val patients")
    return train, val
