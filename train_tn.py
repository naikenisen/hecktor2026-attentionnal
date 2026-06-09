"""Prédiction des stades T et N par RandomForest sur l'embedding TEP/CT figé.

Aucune fusion de données : on n'utilise que l'embedding du bottleneck SwinUNETR
(moyenne + max global), jamais les variables cliniques tabulaires. Un RandomForest
distinct est entraîné pour T et pour N, avec recherche Optuna des hyperparamètres
sur le split de validation (balanced accuracy)."""
import os
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier

import config
from src.image_data import ensure_bottlenecks
from src.clinical_data import load_embeddings
from utils.metrics import balanced_accuracy


def _build_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight="balanced",   # compense le déséquilibre de classes (ex. N2 sur-représenté)
        random_state=config.rf_seed,
        n_jobs=-1,
        **params,
    )


def train_rf(field: str, train, val, n_trials: int, save_path: str) -> float:
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


def main():
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ensure_bottlenecks(config)
    train, val = load_embeddings(config)

    bal_t = train_rf("t_label", train, val, config.tn_n_trials, config.best_tn_t_path)
    bal_n = train_rf("n_label", train, val, config.tn_n_trials, config.best_tn_n_path)
    print(f"\nT balanced accuracy {bal_t:.4f} | N balanced accuracy {bal_n:.4f}")


if __name__ == "__main__":
    main()
