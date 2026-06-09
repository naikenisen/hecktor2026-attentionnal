"""Phase 2 — survie sans rechute par RandomSurvivalForest sur les données cliniques.

Lancement (depuis la racine du dépôt) :

    python -m survival.train

Aucune information image : la forêt de survie est entraînée uniquement sur les variables
tabulaires du CSV (âge standardisé + one-hot : genre, tabac, alcool, performance status,
statut HPV, traitement). T-stage / N-stage sont exclus (ce sont les cibles de `tn`,
indisponibles à l'inférence). Cible : la paire (événement, RFS). La forêt et la recherche
Optuna (c-index) sont mutualisées dans `src.survival_forest`.
"""
import os
import joblib
from sksurv.util import Surv

import config
from src.clinical_data import load_clinical_survival
from src.survival_forest import search_rsf


def main():
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train, val = load_clinical_survival(config)

    X_train, t_train, e_train = train.survival()
    X_val, t_val, e_val = val.survival()
    print(f"survival: {len(t_train)} train / {len(t_val)} val patients with RFS")
    y_train = Surv.from_arrays(event=e_train, time=t_train)

    best, c, params = search_rsf(X_train, y_train, X_val, t_val, e_val, config.surv_n_trials)
    joblib.dump(best, config.best_survival_path)
    print(f"\nbest c-index {c:.4f}")
    print(f"best params {params}")
    print(f"model saved to {config.best_survival_path}")


if __name__ == "__main__":
    main()
