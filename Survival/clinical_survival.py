import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from lifelines.utils import concordance_index
import joblib

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
ALL_FEATURES = [AGE_COLUMN] + BASE_COLUMNS

PARAM_GRID = {
    "rf__n_estimators":    [300, 600],
    "rf__max_depth":       [5, 10, 20, None],
    "rf__max_features":    ["sqrt", "log2"],
    "rf__min_samples_leaf":[4, 8],
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


# Nettoyage du CSV source
raw = pd.read_csv(CSV_PATH)
clean = raw.drop(columns=["T-stage", "N-stage"]).dropna().reset_index(drop=True)
clean.to_csv(CLEAN_CSV_PATH, index=False)

df = pd.read_csv(CLEAN_CSV_PATH)
train = df[df["split"] == "train"]
test  = df[df["split"] == "test"]

X_train = train[ALL_FEATURES]
X_test  = test[ALL_FEATURES]
y_train = Surv.from_arrays(event=train["Relapse"].astype(bool), time=train["RFS"])

# Pipeline complet : préprocessing + modèle dans un seul objet
preproc = ColumnTransformer([
    ("age", StandardScaler(),                        [AGE_COLUMN]),
    ("cat", OneHotEncoder(handle_unknown="ignore"),  BASE_COLUMNS),
])
pipe = Pipeline([
    ("preproc", preproc),
    ("rf",      RandomSurvivalForest(random_state=RF_SEED, n_jobs=-1)),
])

folds = list(
    StratifiedKFold(3, shuffle=True, random_state=RF_SEED)
    .split(X_train, train["Relapse"].astype(bool))
)
search = GridSearchCV(
    pipe, PARAM_GRID,
    cv=folds, error_score=0.0, refit=True, n_jobs=1,
)
search.fit(X_train, y_train)
model = search.best_estimator_
print(f"Best hyperparameters: {search.best_params_}")

t_train, e_train = train["RFS"].to_numpy(float), train["Relapse"].to_numpy(float)
t_test,  e_test  = test["RFS"].to_numpy(float),  test["Relapse"].to_numpy(float)
c_train   = c_index(model.predict(X_train), t_train, e_train)
risk_test = model.predict(X_test)
c_test    = c_index(risk_test, t_test, e_test)
lo, hi    = bootstrap_c_index(risk_test, t_test, e_test)
print(f"train {c_train:.4f}, test {c_test:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

joblib.dump(model, "rfs.joblib")
print("modèle sauvegardé → rfs.joblib")
