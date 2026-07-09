"""Extraction de caractéristiques géométriques des ganglions à partir du seul masque de
segmentation (pas d'apprentissage).

Parcourt un dossier `MASKS_DIR/<PatientID>/<PatientID>.nii.gz` (même arborescence que celle
lue par `seg`) et calcule, par patient, à partir de GTVn (ganglions, label 2) et GTVp (tumeur
primaire, label 1) :

    n_nodes              nombre de ganglions (composantes connexes de GTVn)
    largest_node_mm       plus grand diamètre (mm, coordonnées physiques) parmi les ganglions
                           (0 si aucun ganglion)
    nodes_ipsilateral     True si TOUS les ganglions sont du même côté que la tumeur primaire ;
                           False s'il y a au moins un ganglion controlatéral (ou bilatéral), si
                           le côté de la tumeur est inconnu (pas de GTVp), ou s'il n'y a aucun
                           ganglion

Écrit un CSV `PatientID, n_nodes, largest_node_mm, nodes_ipsilateral`.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label as connected_components
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

MASKS_DIR = "dataset/dataset_masks"                     # dossier contenant un sous-dossier par patient
OUTPUT_CSV = "tables/node_features.csv"
ID_CSV = "tables/n_from_mask.csv"  # CSV de sortie de seg/train.py, pour filtrer les patients connus
GTVP_LABEL = 1                            # tumeur primaire dans le masque HECKTOR
GTVN_LABEL = 2                            # ganglions dans le masque HECKTOR

def merge_csv(id_path, df_features):
    """Fusionne le CSV de caractéristiques géométriques avec le CSV de patients connus."""
    df_ids = pd.read_csv(id_path)
    df_merged = pd.merge(df_ids, df_features, on="PatientID", how="left")
    return df_merged

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

    Sur une affine diagonale (garantie après réorientation canonique) :
    coord(i) = affine[axis, axis] * i + affine[axis, 3].
    """
    return -affine[axis, 3] / affine[axis, axis]


def tumor_side(gtvp: np.ndarray, affine: np.ndarray):
    """Côté de la tumeur primaire (True/False = deux côtés du plan médian sagittal), ou None si absente."""
    if gtvp.sum() == 0:
        return None
    axis = lr_axis(affine)
    midline = midline_index(affine, axis)
    return np.argwhere(gtvp)[:, axis].mean() < midline


def node_features(segmentation_array: np.ndarray, affine: np.ndarray) -> dict:
    """Caractéristiques géométriques des ganglions (GTVn) d'un masque de segmentation.

    Paramètres
    ----------
    segmentation_array : np.ndarray
        Masque de segmentation (entiers), réorienté canonique (RAS), avec GTVP_LABEL=1 pour
        la tumeur primaire et GTVN_LABEL=2 pour les ganglions.
    affine : np.ndarray
        Matrice affine NIfTI (4×4) du volume (image réorientée canonique correspondante).

    Retourne
    --------
    dict : {"n_nodes": int, "largest_node_mm": float, "nodes_ipsilateral": bool}
    """
    gtvn = segmentation_array == GTVN_LABEL
    gtvp = segmentation_array == GTVP_LABEL

    if gtvn.sum() == 0:
        return {"n_nodes": 0, "largest_node_mm": 0.0, "nodes_ipsilateral": False}

    components, n_nodes = connected_components(gtvn)
    axis = lr_axis(affine)
    midline = midline_index(affine, axis)
    t_side = tumor_side(gtvp, affine)

    largest_node_mm = 0.0
    sides = set()
    for k in range(1, n_nodes + 1):
        node = components == k
        largest_node_mm = max(largest_node_mm, max_diameter_mm(node, affine))
        sides.add(np.argwhere(node)[:, axis].mean() < midline)

    nodes_ipsilateral = t_side is not None and sides == {t_side}
    return {
        "n_nodes": n_nodes,
        "largest_node_mm": largest_node_mm,
        "nodes_ipsilateral": nodes_ipsilateral,
    }


if __name__ == "__main__":
    rows = []
    for patient_id in sorted(os.listdir(MASKS_DIR)):
        mask_path = os.path.join(MASKS_DIR, patient_id, f"{patient_id}.nii.gz")
        if not os.path.isfile(mask_path):
            continue
        img = nib.as_closest_canonical(nib.load(mask_path))   # RAS canonique : axe 0 = G→D garanti
        data = np.asarray(img.dataobj)

        feats = node_features(data, img.affine)
        rows.append({"PatientID": patient_id, **feats})
        print(f"{patient_id}: {feats['n_nodes']} ganglion(s), "
              f"plus grand {feats['largest_node_mm']:.1f} mm, "
              f"ipsilatéral={feats['nodes_ipsilateral']}")

    df = pd.DataFrame(rows)
    df = merge_csv(ID_CSV, df)  # ne garder que les patients connus de seg/train.py
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{len(df)} patients → {OUTPUT_CSV}")
