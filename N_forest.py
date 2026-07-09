"""Classifieur RandomForest du stade N à partir des caractéristiques géométriques des
ganglions extraites du masque (`TN.N_features`) — alternative apprise à la règle codée en dur
de `TN/N.py`.

Features (colonnes de `tables/node_features.csv`, calculées par `TN.N_features`) :
    n_nodes             nombre de ganglions (composantes connexes de GTVn)
    largest_node_mm      plus grand diamètre (mm) parmi les ganglions
    nodes_ipsilateral    tous les ganglions du même côté que la tumeur primaire

Cible : `N` (colonne de `tables/node_features.csv`). Ce CSV n'a pas de colonne `split` : le
partage train/test est fait ici via `train_test_split` (stratifié, seed `config.rf_seed`).
Hyperparamètres ajustés par GridSearchCV (StratifiedKFold sur le train uniquement). Écrit la
matrice de confusion (test) dans `figures/`.

Lancement (depuis la racine du dépôt, après `python -m TN.N_features`) :

    python -m TN.N_forest
"""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

import seg.config as config
from TN.N_features import OUTPUT_CSV as FEATURES_CSV
import joblib

FIG_PATH = "figures/confusion_N_forest.png"
FEATURE_COLUMNS = ["n_nodes", "largest_node_mm", "nodes_ipsilateral"]
TARGET_COLUMN = "N"
TEST_SIZE = 0.2

PARAM_GRID = {
    "n_estimators": [50, 200, 500],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 2, 4],
}


def load_dataset() -> pd.DataFrame:
    """Charge les features géométriques et le stade N (`tables/node_features.csv`)."""
    df = pd.read_csv(FEATURES_CSV).dropna(subset=[TARGET_COLUMN])
    df["nodes_ipsilateral"] = df["nodes_ipsilateral"].astype(int)
    return df


def plot_confusion(y_true, y_pred, labels, title, path):
    """Matrice de confusion annotée (vérité en lignes, prédiction en colonnes)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 1, 1.4 * len(labels) + 1))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Prédiction (RandomForest)")
    ax.set_ylabel("Vérité terrain")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"matrice de confusion -> {path}")


def main():
    df = load_dataset()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=config.rf_seed)

    
    model = RandomForestClassifier(max_depth=3, min_samples_leaf=1, n_estimators=50, random_state=config.rf_seed, n_jobs=-1)
    model.fit(X_train, y_train)

    train_acc = balanced_accuracy_score(y_train, model.predict(X_train))
    test_acc = balanced_accuracy_score(y_test, model.predict(X_test))
    print(f"balanced accuracy — train {train_acc:.4f} · test {test_acc:.4f} "
          f"({len(X_train)} train / {len(X_test)} test patients)")

    labels = sorted(set(y_train) | set(y_test))
    plot_confusion(y_test, model.predict(X_test), labels,
                   f"Stade N (RandomForest, features masque) — bal.acc {test_acc:.3f}", FIG_PATH)
    
    joblib.dump(model, "n-staging.joblib")
    print("modèle sauvegardé → n-staging.joblib")


if __name__ == "__main__":
    main()
