"""Prédiction des stades T et N à partir du seul masque de segmentation (pas d'apprentissage).

Parcourt un dossier `MASKS_DIR/<PatientID>/<PatientID>.nii.gz` (même arborescence que celle
lue par `seg`) et mesure la géométrie des structures annotées (GTVp = tumeur primaire = label 1,
GTVn = ganglions = label 2) en coordonnées physiques (mm).

Stade T — diamètre maximal 3D de la tumeur primaire :
    no tumor  = T0
    ≤ 2 cm    = T1
    2 cm < d ≤ 4 cm = T2
    > 4 cm    = T3
    IF HPV status = 1 and > 4 cm = T4

Stade N — ganglions (composantes connexes de GTVn) :
    aucun ganglion                                    → N0
    ganglions ipsilatéraux (même côté que la tumeur)  → N1
    ganglions controlatéraux ou bilatéraux            → N2
    un ganglion > 6 cm                                → N3   (uni ou bilatéral)

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


def tumor_side(gtvp: np.ndarray, affine: np.ndarray):
    """Côté de la tumeur primaire (True/False = deux côtés du plan médian sagittal), ou None si absente."""
    if gtvp.sum() == 0:
        return None
    lr_axis = next(i for i, c in enumerate(nib.aff2axcodes(affine)) if c in "LR")
    midline = gtvp.shape[lr_axis] / 2
    return np.argwhere(gtvp)[:, lr_axis].mean() < midline


def n_stage(gtvn: np.ndarray, affine: np.ndarray, t_side) -> str:
    """Stade N d'après les ganglions : latéralité (relative à la tumeur) et taille.

    N3 si un ganglion > 6 cm ; sinon N1 si tous les ganglions sont ipsilatéraux
    (même côté que la tumeur), N2 s'il y en a de controlatéraux ou des deux côtés.
    Si le côté de la tumeur est inconnu (pas de GTVp), on retombe sur bilatéral → N2.
    """
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


# Statut HPV par patient (clinique) : nécessaire pour distinguer T3 de T4.
hpv_status = pd.read_csv(TRUTH_CSV).set_index("PatientID")["HPV Status"].to_dict()

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
    elif hpv_status.get(patient_id) == 1:    # > 4 cm ET HPV positif → T4
        t_stage = "T4"
    else:
        t_stage = "T3"
    n = n_stage(data == GTVN_LABEL, img.affine, tumor_side(gtvp, img.affine))

    rows.append({"PatientID": patient_id, "T": t_stage, "N": n})
    print(f"{patient_id}: tumeur {diameter:.1f} mm → {t_stage} | {n}")

pred = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
pred.to_csv(OUTPUT_CSV, index=False)
print(f"\n{len(pred)} patients → {OUTPUT_CSV}")

# Comparaison à la vérité terrain (T-stage, N-stage).
# T4 distingué de T3 via le statut HPV ; classes évaluées : T0..T4 et N0..N3.
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
