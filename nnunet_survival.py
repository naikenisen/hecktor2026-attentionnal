"""Phase 2 — survie sans rechute par RandomSurvivalForest sur le bottleneck nnU-Net.

Même source d'embedding que `tn` (features de `tables/bottleneck.csv`, produites par
`seg.extract`), mais cible de survie (événement, RFS) au lieu des stades T/N.
"""
import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from lifelines.utils import concordance_index

CSV_PATH = "tables/HECKTOR_2026_training_data.csv"
BOTTLENECK_CSV_PATH = "tables/bottleneck.csv"
RF_SEED = 42


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


def load_data():
    patients = pd.read_csv(CSV_PATH)
    patients["PatientID"] = patients["PatientID"].astype(str)

    bottleneck = pd.read_csv(BOTTLENECK_CSV_PATH)
    bottleneck["PatientID"] = bottleneck["PatientID"].astype(str)

    patient_info = patients.set_index("PatientID")

    train_rows = bottleneck[bottleneck["PatientID"].map(patient_info["split"]) == "train"]
    test_rows = bottleneck[bottleneck["PatientID"].map(patient_info["split"]) == "test"]

    feature_cols = [c for c in bottleneck.columns if c.startswith("feat_")]

    def process_split(rows):
        X, time, event = [], [], []
        missing_id = 0
        missing_rfs = 0
        for _, row in rows.iterrows():
            pid = row["PatientID"]
            if pid not in patient_info.index:
                missing_id += 1
                continue
            p_row = patient_info.loc[pid]
            rfs = p_row.get("RFS")
            if pd.isna(rfs) or float(rfs) <= 0:
                missing_rfs += 1
                continue
            X.append(row[feature_cols].to_numpy(dtype=np.float32))
            time.append(float(rfs))
            relapse = p_row.get("Relapse")
            event.append(int(relapse) if not pd.isna(relapse) else 0)

        if missing_id or missing_rfs:
            print(f"  survival_xy: {len(time)}/{len(rows)} patients retenus "
                  f"({missing_id} case_id hors CSV, {missing_rfs} sans RFS valide)")
        return np.stack(X) if X else np.empty((0, len(feature_cols)), dtype=np.float32), np.array(time, dtype=np.float64), np.array(event, dtype=bool)

    X_train, t_train, e_train = process_split(train_rows)
    X_test, t_test, e_test = process_split(test_rows)

    print(f"nnU-Net survival: {len(t_train)} train / {len(t_test)} test patients with RFS (dim {X_train.shape[1]})")
    return (X_train, t_train, e_train), (X_test, t_test, e_test)


def main():
    (X_train, t_train, e_train), (X_test, t_test, e_test) = load_data()
    y_train = Surv.from_arrays(event=e_train, time=t_train)

    _, c, params = search_rsf(X_train, y_train, X_test, t_test, e_test)
    print(f"\nbest c-index {c:.4f}")
    print(f"best params {params}")


if __name__ == "__main__":
    main()
