import os
import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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
TN_COLUMNS = ["T-stage", "N-stage"]

PARAM_GRID = {
    "rsf__n_estimators": [300, 600],
    "rsf__max_depth": [5, 10, 20, None],
    "rsf__max_features": ["sqrt", "log2"],
    "rsf__min_samples_leaf": [4, 8],
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


# === étape 1 : preprocessing ===
raw = pd.read_csv(CSV_PATH)
clean = raw.dropna().reset_index(drop=True)
n_events = int(clean["Relapse"].sum())
os.makedirs(os.path.dirname(CLEAN_CSV_PATH) or ".", exist_ok=True)
clean.to_csv(CLEAN_CSV_PATH, index=False)


# === étape 2 : entraînement ===
df = pd.read_csv(CLEAN_CSV_PATH)
df = df[df["RFS"] > 0]

for label, use_tn in [("without TN", False), ("with TN   ", True)]:
    cat_columns = BASE_COLUMNS + (TN_COLUMNS if use_tn else [])
    feature_columns = [AGE_COLUMN] + cat_columns
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train = Surv.from_arrays(event=train["Relapse"].astype(bool), time=train["RFS"])

    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("age", StandardScaler(), [AGE_COLUMN]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_columns),
        ])),
        ("rsf", RandomSurvivalForest(random_state=RF_SEED, n_jobs=-1)),
    ])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=RF_SEED).split(train, y_train["event"]))
    search = GridSearchCV(pipe, PARAM_GRID, cv=folds, error_score=0.0, refit=True, n_jobs=1)
    search.fit(train[feature_columns], y_train)
    model = search.best_estimator_

    t_train, e_train = train["RFS"].to_numpy(float), train["Relapse"].to_numpy(float)
    t_test, e_test = test["RFS"].to_numpy(float), test["Relapse"].to_numpy(float)
    c_train = c_index(model.predict(train[feature_columns]), t_train, e_train)
    risk_test = model.predict(test[feature_columns])
    c_test = c_index(risk_test, t_test, e_test)
    lo, hi = bootstrap_c_index(risk_test, t_test, e_test)
    print(f"{label} : train {c_train:.4f}  |  test {c_test:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
