"""Phase 2 — survie sans rechute par RandomSurvivalForest sur les données cliniques.

Étape 1 : génération du dataset nettoyé (dropna) → tables/clinical_clean.csv
Étape 2 : entraînement de deux RSF (sans TN / avec TN oracle) et report du c-index test.
Treatment est toujours inclus.
"""
import os
import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from lifelines.utils import concordance_index

CSV_PATH = "tables/HECKTOR_2026_training_data.csv"
CLEAN_CSV_PATH = "tables/clinical_clean.csv"
RF_SEED = 42

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


class _SurvivalStratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self._kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        return self._kf.split(X, y["event"])

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._kf.n_splits


_PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [4, 8],
}


def c_index(risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    finite = np.isfinite(risk_scores) & np.isfinite(times) & np.isfinite(events)
    risk_scores, times, events = risk_scores[finite], times[finite], events[finite]
    if len(times) == 0 or events.sum() == 0:
        return 0.5
    return float(concordance_index(times, -risk_scores, events))


def bootstrap_c_index(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """IC 95% du c-index par bootstrap (percentile method). Retourne (ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(times))
    samples = [
        c_index(risk_scores[s], times[s], events[s])
        for s in (rng.choice(idx, size=len(idx), replace=True) for _ in range(n))
    ]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def search_rsf(X_train, y_train, X_test, t_test, e_test) -> tuple:
    search = GridSearchCV(
        estimator=RandomSurvivalForest(random_state=RF_SEED, n_jobs=-1),
        param_grid=_PARAM_GRID,
        cv=_SurvivalStratifiedKFold(n_splits=3, shuffle=True, random_state=RF_SEED),
        error_score=0.0,
        refit=True,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    test_c = c_index(best.predict(X_test), t_test, e_test.astype(float))
    return best, test_c, search.best_params_


def preprocess() -> pd.DataFrame:
    raw = pd.read_csv(CSV_PATH)
    clean = raw.dropna().reset_index(drop=True)

    n_dropped = len(raw) - len(clean)
    n_events = int(clean["Relapse"].sum())
    n_censored = len(clean) - n_events

    print(f"preprocess : {len(raw)} → {len(clean)} patients "
          f"({n_dropped} dropped for missing values)")
    print(f"            events={n_events}  censored={n_censored}  "
          f"event_rate={n_events/len(clean)*100:.1f}%")

    os.makedirs(os.path.dirname(CLEAN_CSV_PATH) or ".", exist_ok=True)
    clean.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"            saved → {CLEAN_CSV_PATH}")
    return clean


class ClinicalEncoder:
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


def load_clinical_survival(use_tn: bool = False) -> tuple[ClinicalSurvivalDataset, ClinicalSurvivalDataset]:
    patients = pd.read_csv(CLEAN_CSV_PATH)
    train_ids = patients.loc[patients["split"] == "train", "PatientID"].astype(str).tolist()
    test_ids = patients.loc[patients["split"] == "test", "PatientID"].astype(str).tolist()
    train_patients = patients[patients["PatientID"].isin(train_ids)]
    encoder = ClinicalEncoder(use_tn=use_tn).fit(train_patients)
    return (
        ClinicalSurvivalDataset(train_ids, patients, encoder),
        ClinicalSurvivalDataset(test_ids, patients, encoder),
    )


def _run(use_tn: bool):
    train, test = load_clinical_survival(use_tn=use_tn)
    X_train, t_train, e_train = train.survival()
    X_test, t_test, e_test = test.survival()
    y_train = Surv.from_arrays(event=e_train, time=t_train)
    best, c_test, _ = search_rsf(X_train, y_train, X_test, t_test, e_test)
    c_train = c_index(best.predict(X_train), t_train, e_train.astype(float))
    lo, hi = bootstrap_c_index(best.predict(X_test), t_test, e_test)
    return c_train, c_test, lo, hi


def main():
    print("=== étape 1 : preprocessing ===")
    preprocess()

    print("\n=== étape 2 : entraînement ===")
    c_train, c_test, lo, hi = _run(use_tn=False)
    print(f"without TN : train {c_train:.4f}  |  test {c_test:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    c_train, c_test, lo, hi = _run(use_tn=True)
    print(f"with TN    : train {c_train:.4f}  |  test {c_test:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
