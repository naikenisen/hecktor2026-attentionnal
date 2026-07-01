"""Phase 2 — prédiction des stades T et N par RandomForest sur l'embedding TEP/CT figé.

Aucune fusion de données : on n'utilise que l'embedding du bottleneck de l'encodeur nnU-Net
(moyenne + max global), jamais les variables cliniques tabulaires. Un RandomForest distinct
est entraîné pour T et pour N, avec recherche par grille (GridSearchCV) sur le split de test.

Les features sont lues dans `tables/bottleneck.csv` (produit par `seg.extract`) : aucune
extraction ici.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit

CSV_PATH = "tables/HECKTOR_2026_training_data.csv"
BOTTLENECK_CSV_PATH = "tables/bottleneck.csv"
RF_SEED = 42

T_STAGES = ["T1", "T2", "T3", "T4"]
N_STAGES = ["N0", "N1", "N2", "N3"]
UNKNOWN_STAGE = -1

_PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [1, 2, 4],
}


def _stage_index(value, stages: list) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return UNKNOWN_STAGE
    label = str(value).strip().upper()
    if label.startswith("N2"):
        label = "N2"
    return stages.index(label) if label in stages else UNKNOWN_STAGE


def load_data():
    patients = pd.read_csv(CSV_PATH)
    patients["PatientID"] = patients["PatientID"].astype(str)

    bottleneck = pd.read_csv(BOTTLENECK_CSV_PATH)
    bottleneck["PatientID"] = bottleneck["PatientID"].astype(str)

    # Map from PatientID to split, T-stage, N-stage
    patient_info = patients.set_index("PatientID")

    # Filter bottleneck rows based on splits from HECKTOR_2026_training_data.csv
    train_rows = bottleneck[bottleneck["PatientID"].map(patient_info["split"]) == "train"]
    test_rows = bottleneck[bottleneck["PatientID"].map(patient_info["split"]) == "test"]

    feature_cols = [c for c in bottleneck.columns if c.startswith("feat_")]

    X_train = train_rows[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_rows[feature_cols].to_numpy(dtype=np.float32)

    t_train = np.array([_stage_index(patient_info.loc[pid, "T-stage"], T_STAGES) for pid in train_rows["PatientID"]], dtype=np.int64)
    t_test = np.array([_stage_index(patient_info.loc[pid, "T-stage"], T_STAGES) for pid in test_rows["PatientID"]], dtype=np.int64)

    n_train = np.array([_stage_index(patient_info.loc[pid, "N-stage"], N_STAGES) for pid in train_rows["PatientID"]], dtype=np.int64)
    n_test = np.array([_stage_index(patient_info.loc[pid, "N-stage"], N_STAGES) for pid in test_rows["PatientID"]], dtype=np.int64)

    print(f"loaded {len(X_train)} train and {len(X_test)} test embeddings (dim {X_train.shape[1]})")
    return X_train, t_train, n_train, X_test, t_test, n_test


def train_rf(field: str, X_train, y_train, X_test, y_test) -> float:
    # Filter out unknown labels (< 0)
    train_mask = y_train >= 0
    test_mask = y_test >= 0

    X_tr, y_tr = X_train[train_mask], y_train[train_mask]
    X_te, y_te = X_test[test_mask], y_test[test_mask]

    print(f"[{field}] {len(y_tr)} train / {len(y_te)} test labelled samples")
    if len(y_tr) == 0 or len(y_te) == 0:
        print(f"[{field}] pas assez de labels — ignoré")
        return 0.0

    # PredefinedSplit : le train reçoit -1, le test reçoit 0.
    X = np.concatenate([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    test_fold = np.concatenate([np.full(len(y_tr), -1), np.zeros(len(y_te))])

    clf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RF_SEED,
        n_jobs=-1,
    )

    search = GridSearchCV(
        estimator=clf,
        param_grid=_PARAM_GRID,
        scoring="balanced_accuracy",
        cv=PredefinedSplit(test_fold),
        refit=False,
        n_jobs=1,
    )
    search.fit(X, y)

    print(f"[{field}] best balanced accuracy {search.best_score_:.4f}")
    print(f"[{field}] best params {search.best_params_}")
    return float(search.best_score_)


def main():
    X_train, t_train, n_train, X_test, t_test, n_test = load_data()

    bal_t = train_rf("t_label", X_train, t_train, X_test, t_test)
    bal_n = train_rf("n_label", X_train, n_train, X_test, n_test)
    print(f"\nT balanced accuracy {bal_t:.4f} | N balanced accuracy {bal_n:.4f}")


if __name__ == "__main__":
    main()
