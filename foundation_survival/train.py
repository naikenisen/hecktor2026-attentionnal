"""Phase 2 (variante) — survie par RandomSurvivalForest sur l'embedding CT-FM (CT seule).

Lancement (depuis la racine du dépôt) :

    pip install lighter_zoo        # dépendance du modèle de fondation
    python -m foundation_survival.train

Contrairement à `survival` (variables cliniques) et `tn` (embedding TEP/CT de la segmentation),
cette tête n'utilise QUE la CT, encodée par le modèle de fondation CT-FM. Le split train/val
réutilise exactement celui de la pipeline image (`split_case_ids`). La forêt et la recherche
aléatoire validée par CV (c-index) sont mutualisées dans `src.survival_forest`.
"""
import os
import joblib
from sksurv.util import Surv

import config
from foundation_survival.extractor import ensure_ct_fm_features
from foundation_survival.dataset import load_foundation_survival
from src.survival_forest import search_rsf


def main():
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ensure_ct_fm_features(config)
    (X_train, t_train, e_train), (X_val, t_val, e_val) = load_foundation_survival(config)
    y_train = Surv.from_arrays(event=e_train, time=t_train)

    best, c, params = search_rsf(X_train, y_train, X_val, t_val, e_val, config.surv_n_trials)
    joblib.dump(best, config.best_foundation_survival_path)
    print(f"\nbest c-index {c:.4f}")
    print(f"best params {params}")
    print(f"model saved to {config.best_foundation_survival_path}")


if __name__ == "__main__":
    main()
