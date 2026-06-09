"""Prédiction de la survie sans rechute par RandomSurvivalForest sur les données cliniques.

Aucune information image : la forêt de survie est entraînée uniquement sur les variables
tabulaires du CSV (âge standardisé + one-hot des variables catégorielles : genre, tabac,
alcool, performance status, statut HPV, traitement). T-stage / N-stage sont exclus (ce sont
les cibles de train_tn, indisponibles à l'inférence). Cible : la paire (événement, RFS).
Recherche Optuna des hyperparamètres sur le c-index de validation."""
import os
import joblib
import optuna
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

import config
from src.clinical_data import load_clinical_survival
from utils.metrics import c_index


def _build_rsf(params: dict) -> RandomSurvivalForest:
    return RandomSurvivalForest(
        random_state=config.rf_seed,
        n_jobs=-1,
        **params,
    )


def main():
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train, val = load_clinical_survival(config)

    X_train, t_train, e_train = train.survival()
    X_val, t_val, e_val = val.survival()
    print(f"survival: {len(t_train)} train / {len(t_val)} val patients with RFS")
    y_train = Surv.from_arrays(event=e_train, time=t_train)

    def objective(trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 15),
        )
        rsf = _build_rsf(params)
        rsf.fit(X_train, y_train)
        # RSF.predict renvoie un score de risque (croissant avec le risque) :
        # c_index attend un score de risque et applique lui-même la négation.
        return c_index(rsf.predict(X_val), t_val, e_val.astype(float))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.rf_seed),
    )
    study.optimize(objective, n_trials=config.surv_n_trials)

    best = _build_rsf(study.best_params)
    best.fit(X_train, y_train)
    joblib.dump(best, config.best_survival_path)
    print(f"\nbest c-index {study.best_value:.4f}")
    print(f"best params {study.best_params}")
    print(f"model saved to {config.best_survival_path}")


if __name__ == "__main__":
    main()
