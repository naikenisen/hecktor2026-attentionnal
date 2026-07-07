"""Prédiction du stade T à partir du seul masque de segmentation (pas d'apprentissage).

Parcourt un dossier `MASKS_DIR/<PatientID>/<PatientID>.nii.gz` (même arborescence que celle
lue par `seg`) et mesure la géométrie de la tumeur primaire (GTVp = label 1) en coordonnées
physiques (mm).

Stade T — diamètre maximal 3D de la tumeur primaire :
    no tumor  = T0
    ≤ 2 cm    = T1
    2 cm < d ≤ 4 cm = T2
    > 4 cm    = T3

Écrit un CSV `PatientID, T`.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.metrics import confusion_matrix

MASKS_DIR = "dataset/dataset_masks"                     # dossier contenant un sous-dossier par patient
OUTPUT_CSV = "tables/t_from_mask.csv"
TRUTH_CSV = "tables/HECKTOR_2026_training_data.csv"     # vérité terrain (T-stage)
FIG_DIR = "figures"                                     # matrice de confusion (PNG)
GTVP_LABEL = 1                            # tumeur primaire dans le masque HECKTOR


def max_diameter_mm(mask: np.ndarray, affine: np.ndarray) -> float:
    """Plus grande distance (mm) entre deux voxels de la structure, en coordonnées physiques."""
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return 0.0
    points = nib.affines.apply_affine(affine, coords)
    if len(points) > 3:
        try:
            points = points[ConvexHull(points).vertices]  # ne garder que l'enveloppe
        except Exception:
            pass
    return float(pdist(points).max())


def plot_confusion(y_true, y_pred, labels, title, path):
    """Matrice de confusion annotée (vérité en lignes, prédiction en colonnes)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 1, 1.4 * len(labels) + 1))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Prédiction (masque)")
    ax.set_ylabel("Vérité terrain")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"matrice de confusion → {path}")


rows = []
for patient_id in sorted(os.listdir(MASKS_DIR)):
    mask_path = os.path.join(MASKS_DIR, patient_id, f"{patient_id}.nii.gz")
    if not os.path.isfile(mask_path):
        continue
    img = nib.load(mask_path)
    data = np.asarray(img.dataobj)

    gtvp = data == GTVP_LABEL
    diameter = max_diameter_mm(gtvp, img.affine)
    if gtvp.sum() == 0:
        t_stage = "T0"                       # pas de tumeur primaire annotée
    elif diameter <= 20:
        t_stage = "T1"
    elif diameter <= 40:
        t_stage = "T2"
    else:
        t_stage = "T3"

    rows.append({"PatientID": patient_id, "T": t_stage})
    print(f"{patient_id}: tumeur {diameter:.1f} mm → {t_stage}")

pred = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
pred.to_csv(OUTPUT_CSV, index=False)
print(f"\n{len(pred)} patients → {OUTPUT_CSV}")

# Comparaison à la vérité terrain (T-stage). Classes évaluées : T0..T3.
truth = pd.read_csv(TRUTH_CSV)[["PatientID", "T-stage"]]
merged = pred.merge(truth, on="PatientID", how="inner")
os.makedirs(FIG_DIR, exist_ok=True)
labelled = merged.dropna(subset=["T-stage"])
accuracy = (labelled["T"] == labelled["T-stage"]).mean()
print(f"T accuracy {accuracy:.4f} ({len(labelled)} patients labellisés)")
# union des classes prédites et réelles, triées, pour une matrice lisible
classes = sorted(set(labelled["T"]) | set(labelled["T-stage"]))
plot_confusion(labelled["T-stage"], labelled["T"], classes,
               f"Stade T (masque) — acc {accuracy:.3f}", os.path.join(FIG_DIR, "confusion_T.png"))
