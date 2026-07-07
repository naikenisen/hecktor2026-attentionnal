"""Prédiction du stade T à partir du seul masque de segmentation (pas d'apprentissage).

Parcourt un dossier `MASKS_DIR/<PatientID>/<PatientID>.nii.gz` (même arborescence que celle
lue par `seg`) et mesure la géométrie de la tumeur primaire (GTVp = label 1) en coordonnées
physiques (mm).

Stade T — présence puis diamètre maximal de la plus grande tumeur primaire. Le diamètre est
mesuré à la convention radiologique : plus grande dimension *dans une coupe axiale* (et non la
diagonale 3D du volume, qui sur-estime la taille et fait migrer les stades vers le haut).

    aucun voxel de tumeur     = T0
    ≥ 1 voxel, d ≤ T1_MAX     = T1
    T1_MAX < d ≤ T2_MAX       = T2
    d > T2_MAX                = T3

Les seuils T1_MAX / T2_MAX ne sont pas fixés à la main : ils sont ajustés par recherche
exhaustive (grid-search) sur les patients labellisés pour maximiser l'accuracy T1/T2/T3.
NB : ajustement *in-sample* (pas de jeu de validation) — à recalibrer en CV pour une mesure
non optimiste. Le stade T4 (invasion de structures adjacentes) n'est pas atteignable par la
taille seule et n'est jamais prédit.

Plusieurs tumeurs primaires : la GTVp est découpée en composantes connexes, mais deux amas
séparés d'un trou < GAP_MM sont fusionnés (segmentation incomplète). Le stade est décidé sur
la plus grande composante ainsi obtenue.

Écrit un CSV `PatientID, T` + une matrice de confusion et une planche 3D de 20 patients
(volume GTVp entier, une couleur par composante) pour le contrôle visuel du seuil de fusion.
"""
import os
import random
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (active la projection '3d')
from scipy.ndimage import label as connected_components, binary_dilation
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.metrics import confusion_matrix

MASKS_DIR = "dataset/dataset_masks"                     # dossier contenant un sous-dossier par patient
OUTPUT_CSV = "tables/t_from_mask.csv"
TRUTH_CSV = "tables/HECKTOR_2026_training_data.csv"     # vérité terrain (T-stage)
FIG_DIR = "figures"                                     # matrice de confusion (PNG)
GTVP_LABEL = 1                            # tumeur primaire dans le masque HECKTOR
GAP_MM = 15.0                             # deux amas séparés de moins que ça = une seule tumeur
N_SAMPLE = 20                             # patients tracés pour le contrôle des composantes
SAMPLE_SEED = 0                           # graine du tirage aléatoire (reproductible)
T1_GRID = np.arange(10, 41)               # seuils T1/T2 candidats (mm) pour le grid-search
T2_GRID = np.arange(20, 81)               # seuils T2/T3 candidats (mm)


def slice_axis(affine: np.ndarray) -> int:
    """Axe du tableau le plus aligné avec l'axe cranio-caudal (Z monde) = axe des coupes axiales."""
    return int(np.argmax(np.abs(affine[2, :3])))


def max_diameter_2d_mm(points: np.ndarray) -> float:
    """Plus grande distance (mm) entre deux points 2D (enveloppe convexe pour borner le coût)."""
    if len(points) < 2:
        return 0.0
    if len(points) > 3:
        try:
            points = points[ConvexHull(points).vertices]
        except Exception:
            pass
    return float(pdist(points).max())


def max_axial_diameter_mm(mask: np.ndarray, affine: np.ndarray) -> float:
    """Plus grande dimension (mm) mesurée *dans une coupe axiale* (convention radiologique).

    Pour chaque coupe le long de l'axe cranio-caudal, on mesure le diamètre in-plane en
    coordonnées physiques, puis on garde le maximum sur toutes les coupes.
    """
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return 0.0
    ax = slice_axis(affine)
    pts = nib.affines.apply_affine(affine, coords)
    drop = int(np.argmax(np.abs(affine[:3, ax])))       # axe monde ~ normale de coupe → écarté
    keep = [i for i in range(3) if i != drop]           # 2 axes monde in-plane
    best = 0.0
    for s in np.unique(coords[:, ax]):
        sub = pts[coords[:, ax] == s][:, keep]
        best = max(best, max_diameter_2d_mm(sub))
    return best


def tumor_components(gtvp: np.ndarray, spacing) -> tuple:
    """Étiquette les tumeurs primaires en tolérant des trous < GAP_MM (segmentation coupée).

    On dilate le masque de ~GAP_MM/2 avant l'étiquetage : deux amas séparés d'un trou plus
    petit que GAP_MM se rejoignent et forment une seule composante ; ceux nettement plus
    éloignés restent distincts. Chaque voxel d'origine est ensuite ré-attribué à sa composante.
    Renvoie (labels, n) avec `labels` limité aux voxels réellement segmentés.
    """
    if not gtvp.any():
        return np.zeros_like(gtvp, dtype=int), 0
    r_vox = max(1, int(round(GAP_MM / 2 / min(spacing))))   # rayon de dilatation en voxels
    dilated = binary_dilation(gtvp, iterations=r_vox)
    comp, n = connected_components(dilated)
    return comp * gtvp, n                                    # ne garder que les voxels d'origine


def largest_diameter_mm(gtvp: np.ndarray, affine: np.ndarray, spacing) -> float:
    """Diamètre axial (mm) de la plus grande composante tumorale (fusion des trous < GAP_MM)."""
    labels, n = tumor_components(gtvp, spacing)
    return max((max_axial_diameter_mm(labels == k, affine) for k in range(1, n + 1)), default=0.0)


def stage_of(diameter: float, has_tumor: bool, t1: float, t2: float) -> str:
    """Stade T à partir du diamètre axial et des seuils ajustés (présence → T1 minimum)."""
    if not has_tumor:
        return "T0"
    return "T1" if diameter <= t1 else "T2" if diameter <= t2 else "T3"


def fit_thresholds(diameter, has_tumor, truth) -> tuple:
    """Cherche (T1_MAX, T2_MAX) maximisant l'accuracy T1/T2/T3 par recherche exhaustive.

    T0 = absence de tumeur (indépendant des seuils) ; T4 (invasion) n'est pas atteignable et
    reste compté comme erreur. Renvoie ((t1, t2), accuracy).
    """
    diameter = np.asarray(diameter, dtype=float)
    has_tumor = np.asarray(has_tumor, dtype=bool)
    truth = np.asarray(truth)
    best_acc, best = -1.0, (25.0, 45.0)
    for t1 in T1_GRID:
        for t2 in T2_GRID:
            if t2 <= t1:
                continue
            pred = np.where(~has_tumor, "T0",
                            np.where(diameter <= t1, "T1",
                                     np.where(diameter <= t2, "T2", "T3")))
            acc = (pred == truth).mean()
            if acc > best_acc:
                best_acc, best = acc, (float(t1), float(t2))
    return best, best_acc


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


def plot_components(patient_ids, t1, t2, path):
    """Planche de contrôle : pour chaque patient, nuage 3D de TOUT le volume GTVp segmenté,
    en coordonnées physiques (mm) et proportions réelles (aucun rescaling), une couleur par
    composante (fusion des trous < GAP_MM). Titre : nb d'amas brut→fusion, diamètre max, stade."""
    cmap = plt.get_cmap("tab10")
    ncol = 5
    nrow = int(np.ceil(len(patient_ids) / ncol))
    fig = plt.figure(figsize=(3.2 * ncol, 3.2 * nrow))
    for idx, patient_id in enumerate(patient_ids, start=1):
        img = nib.load(os.path.join(MASKS_DIR, patient_id, f"{patient_id}.nii.gz"))
        gtvp = np.asarray(img.dataobj) == GTVP_LABEL
        spacing = img.header.get_zooms()[:3]
        n_raw = connected_components(gtvp)[1]            # amas avant fusion des trous < GAP_MM
        labels, n = tumor_components(gtvp, spacing)      # amas après fusion
        diameter = max((max_axial_diameter_mm(labels == k, img.affine)
                        for k in range(1, n + 1)), default=0.0)
        t_stage = stage_of(diameter, gtvp.any(), t1, t2)

        # tous les voxels segmentés → coordonnées physiques (mm), couleur = composante
        coords = np.argwhere(labels > 0)
        pts = nib.affines.apply_affine(img.affine, coords)
        comp_ids = labels[tuple(coords.T)]

        ax = fig.add_subplot(nrow, ncol, idx, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, marker="s",
                   c=[cmap((c - 1) % 10) for c in comp_ids], depthshade=False)
        # proportions réelles : box_aspect = étendue physique de chaque axe (pas de rescaling)
        extent = pts.max(0) - pts.min(0)
        ax.set_box_aspect(np.where(extent > 0, extent, 1))
        ax.set_title(f"{patient_id}\n{n_raw}→{n} comp. · {diameter:.0f} mm · {t_stage}", fontsize=8)
        ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"planche composantes → {path}")


rows = []
for patient_id in sorted(os.listdir(MASKS_DIR)):
    mask_path = os.path.join(MASKS_DIR, patient_id, f"{patient_id}.nii.gz")
    if not os.path.isfile(mask_path):
        continue
    img = nib.load(mask_path)
    data = np.asarray(img.dataobj)

    gtvp = data == GTVP_LABEL
    if not gtvp.any():
        diameter, n_raw = 0.0, 0                         # aucun voxel de tumeur → T0
    else:
        n_raw = connected_components(gtvp)[1]            # nb d'amas bruts (avant fusion)
        # diamètre axial de la plus grande tumeur (composantes, trous < GAP_MM fusionnés)
        diameter = largest_diameter_mm(gtvp, img.affine, img.header.get_zooms()[:3])

    rows.append({"PatientID": patient_id, "diameter": diameter,
                 "has_tumor": bool(gtvp.any()), "n_raw": n_raw})
    print(f"{patient_id}: plus grande tumeur {diameter:.1f} mm (axial)")

pred = pd.DataFrame(rows)

# Ajustement des seuils sur les patients labellisés (grid-search maximisant l'accuracy).
truth = pd.read_csv(TRUTH_CSV)[["PatientID", "T-stage"]]
merged = pred.merge(truth, on="PatientID", how="inner").dropna(subset=["T-stage"])
(t1, t2), fit_acc = fit_thresholds(merged["diameter"], merged["has_tumor"], merged["T-stage"])
print(f"\nseuils ajustés : T1≤{t1:.0f} mm < T2≤{t2:.0f} mm < T3  (accuracy {fit_acc:.4f}, in-sample)")

# Application des seuils à tous les patients + écriture du CSV.
pred["T"] = [stage_of(d, h, t1, t2) for d, h in zip(pred["diameter"], pred["has_tumor"])]
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
pred[["PatientID", "T"]].to_csv(OUTPUT_CSV, index=False)
print(f"{len(pred)} patients → {OUTPUT_CSV}")

# Comparaison à la vérité terrain (T-stage). Classes évaluées : T0..T4.
os.makedirs(FIG_DIR, exist_ok=True)
labelled = pred.merge(truth, on="PatientID", how="inner").dropna(subset=["T-stage"])
accuracy = (labelled["T"] == labelled["T-stage"]).mean()
print(f"T accuracy {accuracy:.4f} ({len(labelled)} patients labellisés)")
# union des classes prédites et réelles, triées, pour une matrice lisible
classes = sorted(set(labelled["T"]) | set(labelled["T-stage"]))
plot_confusion(labelled["T-stage"], labelled["T"], classes,
               f"Stade T (masque) — acc {accuracy:.3f}", os.path.join(FIG_DIR, "confusion_T.png"))

# Contrôle visuel du seuil de fusion : on privilégie les patients à plusieurs amas bruts
# (ceux où le seuil GAP_MM intervient réellement), puis on complète au hasard.
random.seed(SAMPLE_SEED)
multi = [r["PatientID"] for r in rows if r["n_raw"] > 1]
single = [r["PatientID"] for r in rows if r["n_raw"] == 1]
random.shuffle(multi)
random.shuffle(single)
sample = (multi + single)[:N_SAMPLE]
print(f"planche : {min(len(multi), N_SAMPLE)} patients multi-amas + compléments mono-amas")
plot_components(sample, t1, t2, os.path.join(FIG_DIR, "components_T.png"))





# Function to determine T-stage from mask
def t_from_mask(
    segmentation_array: np.ndarray,
    affine: np.ndarray,
    spacing,
    t1: float = 25.0,
    t2: float = 45.0,
) -> str:
    """Renvoie le stade T à partir du tableau de segmentation.

    Paramètres
    ----------
    segmentation_array : np.ndarray
        Masque de segmentation (entiers) avec GTVP_LABEL=1 pour la tumeur primaire.
    affine : np.ndarray
        Matrice affine NIfTI (4×4) du volume — sert à convertir les coordonnées voxel en mm.
    spacing : tuple[float, float, float]
        Taille de voxel en mm sur chaque axe (img.header.get_zooms()[:3]).
        Utilisée pour le seuil de fusion des trous < GAP_MM.
    t1 : float
        Seuil T1/T2 en mm (diamètre axial ≤ t1 → T1). Défaut : 25 mm.
    t2 : float
        Seuil T2/T3 en mm (t1 < diamètre ≤ t2 → T2, sinon T3). Défaut : 45 mm.

    Retourne
    --------
    str : "T0", "T1", "T2" ou "T3"
    """
    gtvp = segmentation_array == GTVP_LABEL
    diameter = largest_diameter_mm(gtvp, affine, spacing)
    return stage_of(diameter, bool(gtvp.any()), t1, t2)

