"""Correction des annotations T4 à partir des prédictions géométriques de `TN.py`.

`TN.py` ne mesure que la géométrie du masque et plafonne à T3 : il ne peut pas
distinguer un T3 d'un T4 (le T4 = envahissement de structures voisines, invisible
à la seule taille de la tumeur). Ce script repart de ses résultats
(`tables/tn_from_mask.csv`), isole les patients classés **T3**, puis décide pour
chacun T3 vs T4 à l'aide d'une régression logistique (elastic-net) entraînée sur
les embeddings `tables/bottleneck.csv`.

L'accuracy est évaluée en **binaire (T3 vs T4)** sur le split test, parmi les
patients prédits T3 dont la vérité terrain est effectivement T3 ou T4.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

TN_CSV_PATH = "tables/tn_from_mask.csv"                  # prédictions géométriques (PatientID, T, N)
BOTTLENECK_CSV_PATH = "tables/bottleneck.csv"           # embeddings (PatientID, feat_*)
TRUTH_CSV_PATH = "tables/HECKTOR_2026_training_data.csv"  # vérité terrain (T-stage, split)
OUTPUT_CSV = "tables/t4_from_mask.csv"                   # prédictions corrigées (T3 → T3/T4)
SEED = 42
PARAM_GRID = {
    "C": [0.01, 0.03, 0.1, 0.3, 1, 3],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

# Jointure : prédictions géométriques + embeddings + vérité terrain
pred = pd.read_csv(TN_CSV_PATH)
pred["PatientID"] = pred["PatientID"].astype(str)
feats = pd.read_csv(BOTTLENECK_CSV_PATH)
feats["PatientID"] = feats["PatientID"].astype(str)
truth = pd.read_csv(TRUTH_CSV_PATH)[["PatientID", "T-stage", "split"]]
truth["PatientID"] = truth["PatientID"].astype(str)

df = pred.merge(feats, on="PatientID", how="inner").merge(truth, on="PatientID", how="inner")
feature_cols = [c for c in df.columns if c.startswith("feat_")]

# On ne travaille que sur les patients que TN.py a classés T3 : ce sont les seuls
# candidats à une reclassification T4.
t3 = df[df["T"] == "T3"].copy()

# Cible binaire : 1 = T4, 0 = T3. On n'entraîne/évalue que sur les patients dont la
# vérité terrain est réellement T3 ou T4 (les autres sont des erreurs de géométrie
# hors de portée de cette correction).
labelled = t3[t3["T-stage"].isin(["T3", "T4"])].copy()
labelled["y"] = (labelled["T-stage"] == "T4").astype(int)

train = labelled[labelled["split"] == "train"]
test = labelled[labelled["split"] == "test"]
print(f"[T4] {len(train)} train / {len(test)} test parmi {len(t3)} patients prédits T3")
print(f"[T4] répartition train  T3={ (train['y']==0).sum() }  T4={ (train['y']==1).sum() }")
print(f"[T4] répartition test   T3={ (test['y']==0).sum() }  T4={ (test['y']==1).sum() }")

y_train = train["y"].to_numpy()
y_test = test["y"].to_numpy()
scaler = StandardScaler().fit(train[feature_cols])
X_train = scaler.transform(train[feature_cols]).astype(np.float32)
X_test = scaler.transform(test[feature_cols]).astype(np.float32)

# Régression logistique elastic-net (grille C × l1_ratio, CV balanced accuracy)
folds = StratifiedKFold(3, shuffle=True, random_state=SEED)
search = GridSearchCV(
    LogisticRegression(
        penalty="elasticnet", solver="saga", class_weight="balanced",
        max_iter=5000, random_state=SEED,
    ),
    PARAM_GRID, scoring="balanced_accuracy", cv=folds, refit=True, n_jobs=-1,
)
search.fit(X_train, y_train)

y_pred = search.predict(X_test)
test_bal_acc = balanced_accuracy_score(y_test, y_pred)
test_acc = accuracy_score(y_test, y_pred)
print(f"\n[T4] best CV balanced accuracy {search.best_score_:.4f}")
print(f"[T4] best params {search.best_params_}")
print(f"[T4] test binary accuracy          {test_acc:.4f}")
print(f"[T4] test binary balanced accuracy {test_bal_acc:.4f}")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(f"[T4] confusion matrix (rows=truth T3/T4, cols=pred T3/T4)\n{cm}")

# Reclassification finale : tous les patients prédits T3 passent dans le modèle,
# ceux prédits positifs deviennent T4. Le reste du CSV est laissé intact.
X_all_t3 = scaler.transform(t3[feature_cols]).astype(np.float32)
t3_pred = search.predict(X_all_t3)
corrected = df.copy()
corrected.loc[t3.index, "T"] = np.where(t3_pred == 1, "T4", "T3")

out = corrected[["PatientID", "T", "N"]]
out.to_csv(OUTPUT_CSV, index=False)
n_new_t4 = int((t3_pred == 1).sum())
print(f"\n{n_new_t4}/{len(t3)} patients prédits T3 reclassés T4 → {OUTPUT_CSV}")
