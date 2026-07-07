# standardisation (z-score) + PCA + RFS

import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from lifelines.utils import concordance_index

CSV_PATH = "tables/HECKTOR_2026_training_data.csv"
BOTTLENECK_CSV_PATH = "tables/bottleneck.csv"
RF_SEED = 42
PCA_VARIANCE = 0.95  # part de variance conservée par la PCA

PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [4, 8],
}

# Jointure et preprocessing
patients = pd.read_csv(CSV_PATH)
patients["PatientID"] = patients["PatientID"].astype(str)
df = pd.read_csv(BOTTLENECK_CSV_PATH)
df["PatientID"] = df["PatientID"].astype(str)
df = df.merge(patients[["PatientID", "split", "RFS", "Relapse"]], on="PatientID", how="inner")
df = df.dropna(subset=["Relapse"])
feature_cols = [c for c in df.columns if c.startswith("feat_")]
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
print(f"nnU-Net survival: {len(train)} train / {len(test)} test patients ")
y_train = Surv.from_arrays(event=train["Relapse"], time=train["RFS"])

# scaler puis PCA, fittés sur train uniquement, appliqués à train et test.
scaler = StandardScaler().fit(train[feature_cols])
pca = PCA(n_components=PCA_VARIANCE, random_state=RF_SEED).fit(scaler.transform(train[feature_cols]))
X_train = pca.transform(scaler.transform(train[feature_cols])).astype(np.float32)
X_test = pca.transform(scaler.transform(test[feature_cols])).astype(np.float32)
print(f"PCA: {len(feature_cols)} → {pca.n_components_} composantes ({PCA_VARIANCE:.0%} de variance)")

# Training
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
print(f"best c-index {c:.4f}")
print(f"best params {search.best_params_}")