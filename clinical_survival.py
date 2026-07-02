import os
import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [4, 8],
}


def c_index(risk_scores, times, events) -> float:
    finite = np.isfinite(risk_scores) & np.isfinite(times) & np.isfinite(events)
    risk_scores, times, events = risk_scores[finite], times[finite], events[finite]
    if len(times) == 0 or events.sum() == 0:
        return 0.5
    return float(concordance_index(times, -risk_scores, events))


def bootstrap_c_index(risk_scores, times, events, n=1000, seed=42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(times))
    samples = [
        c_index(risk_scores[s], times[s], events[s])
        for s in (rng.choice(idx, size=len(idx), replace=True) for _ in range(n))
    ]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# Preprocessing
raw = pd.read_csv(CSV_PATH)
clean = raw.drop(columns=["T-stage", "N-stage"]).dropna().reset_index(drop=True)
clean.to_csv(CLEAN_CSV_PATH, index=False)


# Training
df = pd.read_csv(CLEAN_CSV_PATH)
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

# scaler/encoder fittés sur train uniquement, appliqués à train et test.
scaler = StandardScaler().fit(train[[AGE_COLUMN]])
encoder = OneHotEncoder(handle_unknown="ignore").fit(train[BASE_COLUMNS])
X_train = np.hstack([scaler.transform(train[[AGE_COLUMN]]), encoder.transform(train[BASE_COLUMNS]).toarray()])
X_test = np.hstack([scaler.transform(test[[AGE_COLUMN]]), encoder.transform(test[BASE_COLUMNS]).toarray()])

y_train = Surv.from_arrays(event=train["Relapse"].astype(bool), time=train["RFS"])

folds = list(StratifiedKFold(3, shuffle=True, random_state=RF_SEED).split(X_train, y_train["event"]))
search = GridSearchCV(
    RandomSurvivalForest(random_state=RF_SEED, n_jobs=-1),
    PARAM_GRID, cv=folds, error_score=0.0, refit=True, n_jobs=1,
)
search.fit(X_train, y_train)
model = search.best_estimator_

t_train, e_train = train["RFS"].to_numpy(float), train["Relapse"].to_numpy(float)
t_test, e_test = test["RFS"].to_numpy(float), test["Relapse"].to_numpy(float)
c_train = c_index(model.predict(X_train), t_train, e_train)
risk_test = model.predict(X_test)
c_test = c_index(risk_test, t_test, e_test)
lo, hi = bootstrap_c_index(risk_test, t_test, e_test)
print(f"train {c_train:.4f}  |  test {c_test:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
