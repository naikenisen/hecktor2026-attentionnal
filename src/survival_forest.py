"""RandomSurvivalForest + recherche aléatoire (RandomizedSearchCV) validée par CV.

Partagé par les trois têtes de survie (variables cliniques, embedding nnU-Net, embedding
CT-FM) : elles ne diffèrent que par la source des features, pas par la forêt ni la grille
d'hyperparamètres.

Sélection par validation croisée K-fold sur le train (score par défaut du RSF = concordance
de Harrell), pas sur un unique split de validation : avec les petites cohortes HECKTOR, le
c-index d'un seul split est trop bruité pour sélectionner sans surajuster ce split. Le split
de validation reste tenu à l'écart de la recherche et ne sert qu'au report d'un c-index
honnête.
"""
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sksurv.ensemble import RandomSurvivalForest

import config
from src.metrics import c_index


class _SurvivalStratifiedKFold:
    """KFold stratifié sur l'indicateur d'événement extrait d'un structured array sksurv.
    Garantit qu'aucun fold ne contient uniquement des patients censurés."""
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self._kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        return self._kf.split(X, y["event"])

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._kf.n_splits

# Même espace que la grille Optuna précédente (~10 000 combinaisons) ; RandomizedSearchCV en
# échantillonne n_iter au hasard, ce qui couvre mieux qu'un grid exhaustif borné en budget.
_PARAM_DISTRIBUTIONS = {
    "n_estimators": list(range(200, 1001, 100)),
    "max_depth": list(range(3, 31)),
    "max_features": ["sqrt", "log2", 0.5],
    "min_samples_leaf": list(range(2, 16)),
}


def search_rsf(X_train, y_train, X_val, t_val, e_val, n_iter: int) -> tuple:
    """Recherche aléatoire des hyperparamètres du RSF, sélectionnés par CV K-fold sur le
    c-index, puis réentraîne le meilleur modèle sur tout le train. Le split de validation,
    hors sélection, fournit le c-index reporté. Retourne (modèle, c-index val, params)."""
    search = RandomizedSearchCV(
        estimator=RandomSurvivalForest(random_state=config.rf_seed, n_jobs=-1),
        param_distributions=_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        # scoring=None → score par défaut du RSF = concordance de Harrell (c-index)
        cv=_SurvivalStratifiedKFold(n_splits=3, shuffle=True, random_state=config.rf_seed),
        error_score=0.0,  # fold sans événement → score 0 plutôt que nan pour ne pas invalider toute la combinaison
        random_state=config.rf_seed,
        refit=True,   # réentraîne le meilleur modèle sur tout le train
        n_jobs=1,     # parallélisme déjà porté par la forêt (n_jobs=-1) : pas de sur-souscription
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    # RSF.predict renvoie un score de risque (croissant avec le risque) :
    # c_index attend un score de risque et applique lui-même la négation.
    val_c = c_index(best.predict(X_val), t_val, e_val.astype(float))
    return best, val_c, search.best_params_
