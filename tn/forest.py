"""RandomForest de classification des stades T/N : construction du classifieur et recherche
par grille (`GridSearchCV`) des hyperparamètres, sélectionnés sur la balanced accuracy du
split test (via `PredefinedSplit` : entraînement sur le train, score sur le test).
Aucun modèle n'est sauvegardé — les artefacts d'un run se limitent au dossier `results`.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import config

# Grille exhaustive (volontairement petite pour rester traçable en recherche exhaustive).
_PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [1, 2, 4],
}


def _build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight="balanced",   # compense le déséquilibre de classes (ex. N2 sur-représenté)
        random_state=config.rf_seed,
        n_jobs=-1,
    )


def train_rf(field: str, train, test) -> float:
    """Recherche par grille du meilleur RandomForest pour un champ de stade (`t_label` ou
    `n_label`), sélectionné sur la balanced accuracy du split test. Renvoie ce meilleur score.
    """
    X_train, y_train = train.labelled(field)
    X_test, y_test = test.labelled(field)
    print(f"[{field}] {len(y_train)} train / {len(y_test)} test labelled samples")
    if len(y_train) == 0 or len(y_test) == 0:
        print(f"[{field}] pas assez de labels — ignoré")
        return 0.0

    # PredefinedSplit : le train reçoit -1 (jamais en test), le split test 0 → GridSearchCV
    # entraîne sur le train et score sur le test (aucune fuite : un seul pli fixe).
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    test_fold = np.concatenate([np.full(len(y_train), -1), np.zeros(len(y_test))])

    search = GridSearchCV(
        estimator=_build_rf(),
        param_grid=_PARAM_GRID,
        scoring="balanced_accuracy",
        cv=PredefinedSplit(test_fold),
        refit=False,   # aucun modèle conservé : on ne lit que le meilleur score et ses params
        n_jobs=1,      # parallélisme déjà porté par la forêt (n_jobs=-1)
    )
    search.fit(X, y)

    print(f"[{field}] best balanced accuracy {search.best_score_:.4f}")
    print(f"[{field}] best params {search.best_params_}")
    return float(search.best_score_)
