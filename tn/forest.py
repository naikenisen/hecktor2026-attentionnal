"""RandomForest de classification des stades T/N : construction du classifieur et
recherche Optuna des hyperparamètres sur la balanced accuracy de validation.
"""
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier

import config
from utils.metrics import balanced_accuracy


def _build_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight="balanced",   # compense le déséquilibre de classes (ex. N2 sur-représenté)
        random_state=config.rf_seed,
        n_jobs=-1,
        **params,
    )


def train_rf(field: str, train, val, n_trials: int, save_path: str) -> float:
    """Recherche Optuna puis réentraînement du meilleur RandomForest pour un champ de stade
    (`t_label` ou `n_label`). Sauvegarde le modèle et renvoie la balanced accuracy de val."""
    X_train, y_train = train.labelled(field)
    X_val, y_val = val.labelled(field)
    print(f"[{field}] {len(y_train)} train / {len(y_val)} val labelled samples")

    def objective(trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        )
        clf = _build_rf(params)
        clf.fit(X_train, y_train)
        return balanced_accuracy(y_val, clf.predict(X_val))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.rf_seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best = _build_rf(study.best_params)
    best.fit(X_train, y_train)
    joblib.dump(best, save_path)
    print(f"[{field}] best balanced accuracy {study.best_value:.4f}")
    print(f"[{field}] best params {study.best_params}")
    print(f"[{field}] model saved to {save_path}")
    return study.best_value
