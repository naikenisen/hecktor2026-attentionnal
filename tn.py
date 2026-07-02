# standardisation (z-score) + logistic elastic-net

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

CSV_PATH = "tables/HECKTOR_2026_training_data.csv"
BOTTLENECK_CSV_PATH = "tables/bottleneck.csv"
SEED = 42
y = "T-stage"
# y = "N-stage"
PARAM_GRID = {
    "C": [0.01, 0.03, 0.1, 0.3, 1, 3],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

# Jointure et préprocessing
patients = pd.read_csv(CSV_PATH)
patients["PatientID"] = patients["PatientID"].astype(str)
df = pd.read_csv(BOTTLENECK_CSV_PATH)
df["PatientID"] = df["PatientID"].astype(str)
df = df.merge(patients[["PatientID", "split", y]], on="PatientID", how="inner")
df = df.dropna(subset=[y])
df = df[df[y] != "T0"]
feature_cols = [c for c in df.columns if c.startswith("feat_")]
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
print(f"[{y}] {len(train)} train / {len(test)} test labelled samples")
y_train = train[y].to_numpy()
y_test = test[y].to_numpy()
scaler = StandardScaler().fit(train[feature_cols])
X_train = scaler.transform(train[feature_cols]).astype(np.float32)
X_test = scaler.transform(test[feature_cols]).astype(np.float32)

# training
folds = StratifiedKFold(3, shuffle=True, random_state=SEED)
search = GridSearchCV(
    LogisticRegression(
        penalty="elasticnet", solver="saga", class_weight="balanced",
        max_iter=5000, random_state=SEED,
    ),
    PARAM_GRID, scoring="balanced_accuracy", cv=folds, refit=True, n_jobs=-1,
)
search.fit(X_train, y_train)
test_score = balanced_accuracy_score(y_test, search.predict(X_test))

print(f"[{y}] best CV balanced accuracy {search.best_score_:.4f}")
print(f"[{y}] best params {search.best_params_}")
print(f"[{y}] test balanced accuracy {test_score:.4f}")
