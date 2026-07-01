"""Phase 2 — prédiction des stades T et N par RandomForest sur l'embedding TEP/CT figé.

Lancement (depuis la racine du dépôt) :

    python -m tn.train

Aucune fusion de données : on n'utilise que l'embedding du bottleneck de l'encodeur nnU-Net
(moyenne + max global), jamais les variables cliniques tabulaires. Un RandomForest distinct
est entraîné pour T et pour N, avec recherche par grille (GridSearchCV) sur le split de test.

Les features sont lues dans `tables/bottleneck.csv` (produit par `seg.extract`) : aucune
extraction ici.
"""
import config
from tn.dataset import load_embeddings
from tn.forest import train_rf


def main():
    train, test = load_embeddings(config)

    bal_t = train_rf("t_label", train, test)
    bal_n = train_rf("n_label", train, test)
    print(f"\nT balanced accuracy {bal_t:.4f} | N balanced accuracy {bal_n:.4f}")


if __name__ == "__main__":
    main()
