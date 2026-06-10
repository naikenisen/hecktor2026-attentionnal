"""Encodage des variables cliniques et jeu de survie sans rechute (RFS).

Deux étapes :
  1. preprocess(config) — génère un CSV nettoyé (dropna) sauvegardé dans config.clinical_clean_csv_path
  2. load_clinical_survival(config, use_tn) — lit le CSV nettoyé, encode, retourne train/test

Treatment est toujours inclus. T/N stage sont optionnels (use_tn=True = borne oracle).
"""
import os
import numpy as np
import pandas as pd

from src.split import split_ids

AGE_COLUMN = "Age"
BASE_COLUMNS = [
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "HPV Status",
    "Treatment",
]
TN_COLUMNS = ["T-stage", "N-stage"]
UNKNOWN_CATEGORY = "Inconnu"


def preprocess(config) -> pd.DataFrame:
    """Étape 1 — charge le CSV brut, supprime toutes les lignes avec au moins une valeur
    manquante, sauvegarde le résultat dans config.clinical_clean_csv_path.
    Retourne le DataFrame nettoyé et affiche un résumé."""
    raw = pd.read_csv(config.csv_path_local)
    clean = raw.dropna().reset_index(drop=True)

    n_dropped = len(raw) - len(clean)
    n_events = int(clean["Relapse"].sum())
    n_censored = len(clean) - n_events

    print(f"preprocess : {len(raw)} → {len(clean)} patients "
          f"({n_dropped} dropped for missing values)")
    print(f"            events={n_events}  censored={n_censored}  "
          f"event_rate={n_events/len(clean)*100:.1f}%")

    os.makedirs(os.path.dirname(config.clinical_clean_csv_path), exist_ok=True)
    clean.to_csv(config.clinical_clean_csv_path, index=False)
    print(f"            saved → {config.clinical_clean_csv_path}")
    return clean


class ClinicalEncoder:
    """Apprend sur le train les statistiques d'âge et les catégories observées, puis
    transforme chaque patient en vecteur numérique (âge standardisé + one-hot)."""

    def __init__(self, use_tn: bool = False):
        self.age_mean = 0.0
        self.age_std = 1.0
        self.category_indices = {}
        self._columns = BASE_COLUMNS + (TN_COLUMNS if use_tn else [])

    def fit(self, patients: pd.DataFrame):
        ages = patients[AGE_COLUMN].to_numpy(dtype=float)
        self.age_mean = float(ages.mean())
        self.age_std = float(ages.std()) if ages.std() > 1e-6 else 1.0
        for column in self._columns:
            values = patients[column].astype(str).unique().tolist()
            if UNKNOWN_CATEGORY not in values:
                values.append(UNKNOWN_CATEGORY)
            self.category_indices[column] = {v: i for i, v in enumerate(sorted(values))}
        return self

    @property
    def output_dim(self) -> int:
        return 1 + sum(len(idx) for idx in self.category_indices.values())

    def encode_row(self, row: pd.Series) -> np.ndarray:
        age = float(row[AGE_COLUMN])
        features = [(age - self.age_mean) / self.age_std]
        for column, indices in self.category_indices.items():
            value = str(row[column])
            position = indices.get(value, indices[UNKNOWN_CATEGORY])
            one_hot = [0.0] * len(indices)
            one_hot[position] = 1.0
            features.extend(one_hot)
        return np.array(features, dtype=np.float32)


class ClinicalSurvivalDataset:
    """Variables cliniques encodées d'un split, alignées par case_id avec (temps, événement)."""

    def __init__(self, case_ids: list, patients: pd.DataFrame, encoder: ClinicalEncoder):
        rows = patients.set_index("PatientID")
        features, time, event, kept = [], [], [], []
        for case_id in case_ids:
            if case_id not in rows.index:
                continue
            row = rows.loc[case_id]
            features.append(encoder.encode_row(row))
            time.append(float(row["RFS"]))
            event.append(int(row["Relapse"]))
            kept.append(case_id)
        self.case_ids = kept
        self.X = np.stack(features) if features else np.empty((0, encoder.output_dim), np.float32)
        self.time = np.array(time, dtype=np.float64)
        self.event = np.array(event, dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def survival(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = self.time > 0
        return self.X[mask], self.time[mask], self.event[mask].astype(bool)


def load_clinical_survival(
    config, use_tn: bool = False
) -> tuple[ClinicalSurvivalDataset, ClinicalSurvivalDataset]:
    """Étape 2 — lit le CSV nettoyé (doit avoir été généré par preprocess()),
    encode et retourne (train, test)."""
    patients = pd.read_csv(config.clinical_clean_csv_path)
    all_ids = patients["PatientID"].astype(str).tolist()
    train_ids, test_ids = split_ids(all_ids, config)
    train_patients = patients[patients["PatientID"].isin(train_ids)]
    encoder = ClinicalEncoder(use_tn=use_tn).fit(train_patients)
    return (
        ClinicalSurvivalDataset(train_ids, patients, encoder),
        ClinicalSurvivalDataset(test_ids, patients, encoder),
    )
