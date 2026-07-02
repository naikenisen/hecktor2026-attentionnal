# standardisation (z-score) + PCA + RFS

import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from lifelines.utils import concordance_index

CSV_PATH = "tables/HECKTOR_2026_training_data.csv"
BOTTLENECK_CSV_PATH = "tables/bottleneck.csv"
RF_SEED = 42

PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [4, 8],
}

# Chargement : jointure bottleneck ↔ CSV patients, filtre sur RFS valide.
patients = pd.read_csv(CSV_PATH)
patients["PatientID"] = patients["PatientID"].astype(str)
df = pd.read_csv(BOTTLENECK_CSV_PATH)
df["PatientID"] = df["PatientID"].astype(str)
df = df.merge(patients[["PatientID", "split", "RFS", "Relapse"]], on="PatientID", how="inner")
df = df[df["RFS"].notna() & (df["RFS"] > 0)]
df["Relapse"] = df["Relapse"].fillna(0).astype(bool)

feature_cols = [c for c in df.columns if c.startswith("feat_")]
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
print(f"nnU-Net survival: {len(train)} train / {len(test)} test patients "
      f"with RFS (dim {len(feature_cols)})")

X_train = train[feature_cols].to_numpy(np.float32)
X_test = test[feature_cols].to_numpy(np.float32)
y_train = Surv.from_arrays(event=train["Relapse"], time=train["RFS"])

# Recherche par grille, folds stratifiés sur l'événement.
folds = list(StratifiedKFold(3, shuffle=True, random_state=RF_SEED).split(X_train, y_train["event"]))
search = GridSearchCV(
    RandomSurvivalForest(random_state=RF_SEED, n_jobs=-1),
    PARAM_GRID, cv=folds, error_score=0.0, refit=True, n_jobs=1,
)
search.fit(X_train, y_train)

risk = search.best_estimator_.predict(X_test)
times, events = test["RFS"].to_numpy(float), test["Relapse"].to_numpy(float)
finite = np.isfinite(risk) & np.isfinite(times) & np.isfinite(events)
c = float(concordance_index(times[finite], -risk[finite], events[finite])) if events[finite].sum() else 0.5
print(f"\nbest c-index {c:.4f}")
print(f"best params {search.best_params_}")
