"""Phase 2 — survie sans rechute par RandomSurvivalForest sur les données cliniques.

Lancement (depuis la racine du dépôt) :

    python -m clinical_survival.train

Étape 1 : génération du dataset nettoyé (dropna) → config.clinical_clean_csv_path
Étape 2 : entraînement de deux RSF (sans TN / avec TN oracle) et report du c-index test.
Treatment est toujours inclus.
"""
import os
import joblib
from sksurv.util import Surv

import config
from clinical_survival.dataset import preprocess, load_clinical_survival
from src.survival_forest import search_rsf
from src.metrics import bootstrap_c_index, c_index


def _run(use_tn: bool, save_path: str):
    """Entraîne le RSF et retourne (c_train, c_test, ci_low, ci_high)."""
    train, test = load_clinical_survival(config, use_tn=use_tn)
    X_train, t_train, e_train = train.survival()
    X_test, t_test, e_test = test.survival()
    y_train = Surv.from_arrays(event=e_train, time=t_train)
    best, c_test, _ = search_rsf(X_train, y_train, X_test, t_test, e_test, config.surv_n_trials)
    joblib.dump(best, save_path)
    c_train = c_index(best.predict(X_train), t_train, e_train.astype(float))
    lo, hi = bootstrap_c_index(best.predict(X_test), t_test, e_test)
    return c_train, c_test, lo, hi


def main():
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print("=== étape 1 : preprocessing ===")
    preprocess(config)

    print("\n=== étape 2 : entraînement ===")
    c_train, c_test, lo, hi = _run(use_tn=False, save_path=config.best_survival_path)
    print(f"without TN : train {c_train:.4f}  |  test {c_test:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    c_train, c_test, lo, hi = _run(use_tn=True, save_path=config.best_survival_tn_path)
    print(f"with TN    : train {c_train:.4f}  |  test {c_test:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
