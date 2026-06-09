"""Encodage des variables cliniques et jeu de survie sans rechute (RFS).

Aucune information image : seulement les variables tabulaires du CSV (âge standardisé +
one-hot). T-stage / N-stage sont exclus (ce sont les cibles de `tn`, indisponibles à
l'inférence). Le split réutilise `src.split.split_case_ids` (cohérent avec les autres têtes).
"""
import numpy as np
import pandas as pd

from src.split import split_case_ids

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
    """Construit les jeux de survie cliniques train/val. Le split (`split_case_ids`) est
    identique à celui de la pipeline image, pour rester cohérent avec `tn`, mais aucune image
    — ni MONAI — n'est chargée ici. L'encodeur est ajusté sur le train."""
    train_ids, val_ids = split_case_ids(config)
    patients = pd.read_csv(config.csv_path)
    train_patients = patients[patients["PatientID"].isin(train_ids)]
    encoder = ClinicalEncoder().fit(train_patients)
    train = ClinicalSurvivalDataset(train_ids, patients, encoder)
    val = ClinicalSurvivalDataset(val_ids, patients, encoder)
    print(f"loaded clinical features (dim {encoder.output_dim}): "
          f"{len(train)} train / {len(val)} val patients")
    return train, val
