import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

CSV_PATH = "tables/HECKTOR_2026_training_data.csv"
BOTTLENECK_CSV_PATH = "tables/bottleneck.csv"
SEED = 42
y = "T-stage"  # ou "N-stage"

PARAM_GRID = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", "auto"],
}

# Jointure et préprocessing
patients = pd.read_csv(CSV_PATH)
patients["PatientID"] = patients["PatientID"].astype(str)
df = pd.read_csv(BOTTLENECK_CSV_PATH)
df["PatientID"] = df["PatientID"].astype(str)
df = df.merge(patients[["PatientID", "split", y]], on="PatientID", how="inner")
df = df.dropna(subset=[y])
feature_cols = [c for c in df.columns if c.startswith("feat_")]
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
print(f"[{y}] {len(train)} train / {len(test)} test labelled samples")
X_train = train[feature_cols].to_numpy(np.float32)
X_test = test[feature_cols].to_numpy(np.float32)
y_train = train[y].to_numpy()
y_test = test[y].to_numpy()

# training
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(class_weight="balanced", random_state=SEED)),
])
folds = StratifiedKFold(3, shuffle=True, random_state=SEED)
search = GridSearchCV(pipe, PARAM_GRID, scoring="balanced_accuracy", cv=folds, refit=True, n_jobs=-1)
search.fit(X_train, y_train)
test_score = balanced_accuracy_score(y_test, search.predict(X_test))

print(f"[{y}] best CV balanced accuracy {search.best_score_:.4f}")
print(f"[{y}] best params {search.best_params_}")
print(f"[{y}] test balanced accuracy {test_score:.4f}")
