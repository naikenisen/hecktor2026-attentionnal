"""Prédiction des stades T et N à partir du seul masque de segmentation (pas d'apprentissage).

Parcourt un dossier `MASKS_DIR/<PatientID>/<PatientID>.nii.gz` (même arborescence que celle
lue par `seg`) et mesure la géométrie des structures annotées (GTVp = tumeur primaire = label 1,
GTVn = ganglions = label 2) en coordonnées physiques (mm).

Stade T — diamètre maximal 3D de la tumeur primaire :
    ≤ 2 cm        → T1
    2 cm < d ≤ 4 cm → T2
    > 4 cm        → T3

Stade N — ganglions (composantes connexes de GTVn) :
    aucun ganglion        → N0
    un ganglion > 6 cm    → N3   (uni ou bilatéral)
    ganglions bilatéraux  → N2
    ganglions unilatéraux → N1

Écrit un CSV `PatientID, T, N`.
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
OUTPUT_CSV = "tables/tn_from_mask.csv"
TRUTH_CSV = "tables/HECKTOR_2026_training_data.csv"     # vérité terrain (T-stage, N-stage)
FIG_DIR = "figures"                                     # matrices de confusion (PNG)
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


def n_stage(gtvn: np.ndarray, affine: np.ndarray) -> str:
    """Stade N d'après les ganglions : latéralité (côté du plan médian sagittal) et taille."""
    if gtvn.sum() == 0:
        return "N0"
    lr_axis = next(i for i, c in enumerate(nib.aff2axcodes(affine)) if c in "LR")
    midline = gtvn.shape[lr_axis] / 2
    components, n = connected_components(gtvn)

    sides, max_node = set(), 0.0
    for k in range(1, n + 1):
        node = components == k
        coords = np.argwhere(node)
        sides.add(coords[:, lr_axis].mean() < midline)   # True/False = deux côtés du plan médian
        max_node = max(max_node, max_diameter_mm(node, affine))

    if max_node > 60:
        return "N3"
    return "N2" if len(sides) > 1 else "N1"


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
    else:
        t_stage = "T1" if diameter <= 20 else "T2" if diameter <= 40 else "T3"
    n = n_stage(data == GTVN_LABEL, img.affine)

    rows.append({"PatientID": patient_id, "T": t_stage, "N": n})
    print(f"{patient_id}: tumeur {diameter:.1f} mm → {t_stage} | {n}")

pred = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
pred.to_csv(OUTPUT_CSV, index=False)
print(f"\n{len(pred)} patients → {OUTPUT_CSV}")

# Comparaison à la vérité terrain (T-stage, N-stage).
# T3/T4 regroupés (la géométrie plafonne à T3) → classes évaluées : T0, T1, T2, T3.
truth = pd.read_csv(TRUTH_CSV)[["PatientID", "T-stage", "N-stage"]]
merged = pred.merge(truth, on="PatientID", how="inner")
os.makedirs(FIG_DIR, exist_ok=True)
for pred_col, truth_col, title, fname in [
    ("T", "T-stage", "Stade T (masque)", "confusion_T.png"),
    ("N", "N-stage", "Stade N (masque)", "confusion_N.png"),
]:
    labelled = merged.dropna(subset=[truth_col])
    accuracy = (labelled[pred_col] == labelled[truth_col]).mean()
    print(f"{pred_col} accuracy {accuracy:.4f} ({len(labelled)} patients labellisés)")
    # union des classes prédites et réelles, triées, pour une matrice lisible
    classes = sorted(set(labelled[pred_col]) | set(labelled[truth_col]))
    plot_confusion(labelled[truth_col], labelled[pred_col], classes,
                   f"{title} — acc {accuracy:.3f}", os.path.join(FIG_DIR, fname))
