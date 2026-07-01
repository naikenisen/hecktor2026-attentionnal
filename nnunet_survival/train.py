"""Phase 2 — survie sans rechute par RandomSurvivalForest sur le bottleneck nnU-Net.

Lancement (depuis la racine du dépôt) :

    python -m nnunet_survival.train

Même source d'embedding que `tn` (bottleneck de l'encodeur nnU-Net, extrait une seule fois
puis mis en cache), mais cible de survie (événement, RFS) au lieu des stades T/N. La forêt
et la recherche par grille validée par CV (c-index) sont mutualisées dans `src.survival_forest`.
"""
from sksurv.util import Surv

import config
from src.nnunet_embedding import ensure_bottlenecks
from nnunet_survival.dataset import load_nnunet_survival
from src.survival_forest import search_rsf


def main():
    ensure_bottlenecks(config)
    (X_train, t_train, e_train), (X_val, t_val, e_val) = load_nnunet_survival(config)
    y_train = Surv.from_arrays(event=e_train, time=t_train)

    _, c, params = search_rsf(X_train, y_train, X_val, t_val, e_val)
    print(f"\nbest c-index {c:.4f}")
    print(f"best params {params}")


if __name__ == "__main__":
    main()
