"""RandomSurvivalForest + recherche Optuna sur le c-index de validation.

Partagé par les deux têtes de survie (variables cliniques et embedding CT-FM) : elles ne
diffèrent que par la source des features, pas par la forêt ni la grille d'hyperparamètres.
"""
import optuna
from sksurv.ensemble import RandomSurvivalForest

import config
from utils.metrics import c_index


def _build_rsf(params: dict) -> RandomSurvivalForest:
    return RandomSurvivalForest(random_state=config.rf_seed, n_jobs=-1, **params)


def search_rsf(X_train, y_train, X_val, t_val, e_val, n_trials: int) -> tuple:
    """Recherche Optuna des hyperparamètres du RSF sur le c-index de validation, puis
    réentraîne le meilleur modèle sur le train. Retourne (modèle, c-index, params)."""
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
    study.optimize(objective, n_trials=n_trials)

    best = _build_rsf(study.best_params)
    best.fit(X_train, y_train)
    return best, study.best_value, study.best_params
