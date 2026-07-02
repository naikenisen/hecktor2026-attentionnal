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

PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [1, 2, 4],
}


def stage_index(values: pd.Series, stages: list) -> pd.Series:
    label = values.astype(str).str.strip().str.upper()
    label = label.where(~label.str.startswith("N2"), "N2")  # N2a/N2b/N2c → N2
    return label.map({s: i for i, s in enumerate(stages)}).fillna(UNKNOWN_STAGE).astype(int)


# Chargement : jointure bottleneck ↔ CSV patients, encodage des stades T/N.
patients = pd.read_csv(CSV_PATH)
patients["PatientID"] = patients["PatientID"].astype(str)
df = pd.read_csv(BOTTLENECK_CSV_PATH)
df["PatientID"] = df["PatientID"].astype(str)
df = df.merge(patients[["PatientID", "split", "T-stage", "N-stage"]], on="PatientID", how="inner")
df["t_label"] = stage_index(df["T-stage"], T_STAGES)
df["n_label"] = stage_index(df["N-stage"], N_STAGES)

feature_cols = [c for c in df.columns if c.startswith("feat_")]
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
print(f"loaded {len(train)} train and {len(test)} test embeddings (dim {len(feature_cols)})")

scores = {}
for field in ("t_label", "n_label"):
    tr = train[train[field] >= 0]
    te = test[test[field] >= 0]
    print(f"[{field}] {len(tr)} train / {len(te)} test labelled samples")
    if len(tr) == 0 or len(te) == 0:
        print(f"[{field}] pas assez de labels — ignoré")
        scores[field] = 0.0
        continue

    # PredefinedSplit : le train reçoit -1, le test reçoit 0.
    X = np.concatenate([tr[feature_cols], te[feature_cols]]).astype(np.float32)
    y = np.concatenate([tr[field], te[field]])
    test_fold = np.concatenate([np.full(len(tr), -1), np.zeros(len(te))])

    search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RF_SEED, n_jobs=-1),
        PARAM_GRID, scoring="balanced_accuracy", cv=PredefinedSplit(test_fold),
        refit=False, n_jobs=1,
    )
    search.fit(X, y)
    print(f"[{field}] best balanced accuracy {search.best_score_:.4f}")
    print(f"[{field}] best params {search.best_params_}")
    scores[field] = float(search.best_score_)

print(f"\nT balanced accuracy {scores['t_label']:.4f} | N balanced accuracy {scores['n_label']:.4f}")
