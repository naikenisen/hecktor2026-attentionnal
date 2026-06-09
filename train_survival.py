"""Prédiction de la survie sans rechute par RandomSurvivalForest sur l'embedding TEP/CT.

Aucune fusion de données : la forêt de survie est entraînée uniquement sur l'embedding
du bottleneck SwinUNETR (moyenne + max global), avec pour cible la paire (événement,
temps de RFS). Recherche Optuna des hyperparamètres sur le c-index de validation."""
import os
import joblib
import optuna
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

import config
from src.image_data import ensure_bottlenecks
from src.clinical_data import load_embeddings
from utils.metrics import c_index


def _build_rsf(params: dict) -> RandomSurvivalForest:
    return RandomSurvivalForest(
        random_state=config.rf_seed,
        n_jobs=-1,
        **params,
    )


def main():
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ensure_bottlenecks(config)
    train, val = load_embeddings(config)

    X_train, t_train, e_train = train.survival()
    X_val, t_val, e_val = val.survival()
    print(f"survival: {len(t_train)} train / {len(t_val)} val patients with RFS")
    y_train = Surv.from_arrays(event=e_train, time=t_train)

    def objective(trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 15),
        )
        rsf = _build_rsf(params)
        rsf.fit(X_train, y_train)
        # RSF.predict renvoie un score de risque (croissant avec le risque) :
        # c_index attend un score de risque et applique lui-même la négation.
        return c_index(rsf.predict(X_val), t_val, e_val.astype(float))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.surv_n_trials)

    best = _build_rsf(study.best_params)
    best.fit(X_train, y_train)
    joblib.dump(best, config.best_survival_path)
    print(f"\nbest c-index {study.best_value:.4f}")
    print(f"best params {study.best_params}")
    print(f"model saved to {config.best_survival_path}")


if __name__ == "__main__":
    main()
