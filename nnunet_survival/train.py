"""Phase 2 — survie sans rechute par RandomSurvivalForest sur le bottleneck nnU-Net.

Lancement (depuis la racine du dépôt) :

    python -m nnunet_survival.train

Même source d'embedding que `tn` (features de `tables/bottleneck.csv`, produites par
`seg.extract`), mais cible de survie (événement, RFS) au lieu des stades T/N. La forêt et la
recherche par grille validée par CV (c-index) sont mutualisées dans `src.survival_forest`.
Aucune extraction ici : on ne fait que lire le CSV.
"""
from sksurv.util import Surv

import config
from nnunet_survival.dataset import load_nnunet_survival
from src.survival_forest import search_rsf


def main():
    (X_train, t_train, e_train), (X_test, t_test, e_test) = load_nnunet_survival(config)
    y_train = Surv.from_arrays(event=e_train, time=t_train)

    _, c, params = search_rsf(X_train, y_train, X_test, t_test, e_test)
    print(f"\nbest c-index {c:.4f}")
    print(f"best params {params}")


if __name__ == "__main__":
    main()
