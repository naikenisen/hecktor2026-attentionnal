"""Prédiction du stade N à partir du seul masque de segmentation (pas d'apprentissage).

Parcourt un dossier `MASKS_DIR/<PatientID>/<PatientID>.nii.gz` (même arborescence que celle
lue par `seg`) et mesure la géométrie des ganglions (GTVn = label 2) en coordonnées
physiques (mm). La latéralité est appréciée par rapport à la tumeur primaire (GTVp = label 1).

Stade N — ganglions (composantes connexes de GTVn) :
    aucun ganglion                                    → N0
    ganglions ipsilatéraux (même côté que la tumeur)  → N1
    ganglions controlatéraux ou bilatéraux            → N2
    un ganglion > 6 cm                                → N3   (uni ou bilatéral)

Écrit un CSV `PatientID, N`.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label as connected_components
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.metrics import confusion_matrix

MASKS_DIR = "dataset/dataset_masks"                     # dossier contenant un sous-dossier par patient
OUTPUT_CSV = "tables/n_from_mask.csv"
TRUTH_CSV = "tables/HECKTOR_2026_training_data.csv"     # vérité terrain (N-stage)
FIG_DIR = "figures"                                     # matrice de confusion (PNG)
GTVP_LABEL = 1                            # tumeur primaire dans le masque HECKTOR
GTVN_LABEL = 2                            # ganglions dans le masque HECKTOR


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


def lr_axis(affine: np.ndarray) -> int:
    """Index de l'axe gauche/droite d'après l'affine NIfTI."""
    return next(i for i, c in enumerate(nib.aff2axcodes(affine)) if c in "LR")


def midline_index(affine: np.ndarray, axis: int) -> float:
    """Indice de voxel du plan médian sagittal (x = 0 mm en coordonnées monde).

    Identique à la logique de viz_sagittal_simple : plutôt que le milieu du
    tableau (shape/2), on prend le voxel dont la coordonnée physique le long de
    l'axe G/D vaut 0. Sur une affine diagonale (garantie après réorientation
    canonique) : coord(i) = affine[axis, axis] * i + affine[axis, 3].
    """
    return -affine[axis, 3] / affine[axis, axis]


def tumor_side(gtvp: np.ndarray, affine: np.ndarray):
    """Côté de la tumeur primaire (True/False = deux côtés du plan médian sagittal), ou None si absente."""
    if gtvp.sum() == 0:
        return None
    axis = lr_axis(affine)
    midline = midline_index(affine, axis)
    return np.argwhere(gtvp)[:, axis].mean() < midline


def n_stage(gtvn: np.ndarray, affine: np.ndarray, t_side) -> str:
    """Stade N d'après les ganglions : latéralité (relative à la tumeur) et taille.

    N3 si un ganglion > 6 cm ; sinon N1 si tous les ganglions sont ipsilatéraux
    (même côté que la tumeur), N2 s'il y en a de controlatéraux ou des deux côtés.
    Si le côté de la tumeur est inconnu (pas de GTVp), on retombe sur bilatéral → N2.
    """
    if gtvn.sum() == 0:
        return "N0"
    axis = lr_axis(affine)
    midline = midline_index(affine, axis)
    components, n = connected_components(gtvn)

    sides, max_node = set(), 0.0
    for k in range(1, n + 1):
        node = components == k
        coords = np.argwhere(node)
        sides.add(coords[:, axis].mean() < midline)   # True/False = deux côtés du plan médian
        max_node = max(max_node, max_diameter_mm(node, affine))

    if max_node > 60:
        return "N3"
    if t_side is None:                       # côté de la tumeur inconnu : bilatéral → N2, sinon N1
        return "N2" if len(sides) > 1 else "N1"
    # ipsilatéral seulement (uniquement le côté de la tumeur) → N1 ; controlatéral ou bilatéral → N2
    return "N1" if sides == {t_side} else "N2"


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
    img = nib.as_closest_canonical(nib.load(mask_path))   # RAS canonique : axe 0 = G→D garanti
    data = np.asarray(img.dataobj)

    gtvp = data == GTVP_LABEL
    n = n_stage(data == GTVN_LABEL, img.affine, tumor_side(gtvp, img.affine))

    rows.append({"PatientID": patient_id, "N": n})
    print(f"{patient_id}: {n}")

pred = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
pred.to_csv(OUTPUT_CSV, index=False)
print(f"\n{len(pred)} patients → {OUTPUT_CSV}")

# Comparaison à la vérité terrain (N-stage). Classes évaluées : N0..N3.
truth = pd.read_csv(TRUTH_CSV)[["PatientID", "N-stage"]]
merged = pred.merge(truth, on="PatientID", how="inner")
os.makedirs(FIG_DIR, exist_ok=True)
labelled = merged.dropna(subset=["N-stage"])
accuracy = (labelled["N"] == labelled["N-stage"]).mean()
print(f"N accuracy {accuracy:.4f} ({len(labelled)} patients labellisés)")
# union des classes prédites et réelles, triées, pour une matrice lisible
classes = sorted(set(labelled["N"]) | set(labelled["N-stage"]))
plot_confusion(labelled["N-stage"], labelled["N"], classes,
               f"Stade N (masque) — acc {accuracy:.3f}", os.path.join(FIG_DIR, "confusion_N.png"))
