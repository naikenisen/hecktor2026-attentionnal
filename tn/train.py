"""Phase 2 — prédiction des stades T et N par RandomForest sur l'embedding TEP/CT figé.

Lancement (depuis la racine du dépôt) :

    python -m tn.train

Aucune fusion de données : on n'utilise que l'embedding du bottleneck de l'encodeur nnU-Net
(moyenne + max global), jamais les variables cliniques tabulaires. Un RandomForest distinct
est entraîné pour T et pour N, avec recherche Optuna sur le split de validation.
"""
import os
import config
from src.image_data import ensure_bottlenecks
from src.clinical_data import load_embeddings
from tn.forest import train_rf


def main():
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ensure_bottlenecks(config)
    train, val = load_embeddings(config)

    bal_t = train_rf("t_label", train, val, config.tn_n_trials, config.best_tn_t_path)
    bal_n = train_rf("n_label", train, val, config.tn_n_trials, config.best_tn_n_path)
    print(f"\nT balanced accuracy {bal_t:.4f} | N balanced accuracy {bal_n:.4f}")


if __name__ == "__main__":
    main()
