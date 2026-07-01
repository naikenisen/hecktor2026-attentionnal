"""RandomSurvivalForest + recherche exhaustive par grille (`GridSearchCV`) validée par CV.

Partagé par les têtes de survie (variables cliniques, embedding nnU-Net) : elles ne diffèrent
que par la source des features, pas par la forêt ni la grille d'hyperparamètres.

Sélection par validation croisée K-fold sur le train (score par défaut du RSF = concordance
de Harrell), pas sur un unique split : avec les petites cohortes HECKTOR, le c-index d'un
seul split est trop bruité pour sélectionner sans surajuster ce split. Le split test reste
tenu à l'écart de la recherche et ne sert qu'au report d'un c-index honnête. Aucun modèle
n'est sauvegardé — les artefacts d'un run se limitent au dossier `results`.
"""
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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


# Grille exhaustive (volontairement petite pour rester traçable) balayée par CV.
_PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [4, 8],
}


def search_rsf(X_train, y_train, X_test, t_test, e_test) -> tuple:
    """Recherche par grille des hyperparamètres du RSF, sélectionnés par CV K-fold sur le
    c-index, puis réentraîne le meilleur modèle sur tout le train. Le split test, hors
    sélection, fournit le c-index reporté. Retourne (modèle, c-index test, params)."""
    search = GridSearchCV(
        estimator=RandomSurvivalForest(random_state=config.rf_seed, n_jobs=-1),
        param_grid=_PARAM_GRID,
        # scoring=None → score par défaut du RSF = concordance de Harrell (c-index)
        cv=_SurvivalStratifiedKFold(n_splits=3, shuffle=True, random_state=config.rf_seed),
        error_score=0.0,  # fold sans événement → score 0 plutôt que nan pour ne pas invalider la combinaison
        refit=True,       # réentraîne le meilleur modèle sur tout le train
        n_jobs=1,         # parallélisme déjà porté par la forêt (n_jobs=-1) : pas de sur-souscription
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    # RSF.predict renvoie un score de risque (croissant avec le risque) :
    # c_index attend un score de risque et applique lui-même la négation.
    test_c = c_index(best.predict(X_test), t_test, e_test.astype(float))
    return best, test_c, search.best_params_
