"""Prédiction du stade T à partir du seul masque de segmentation (pas d'apprentissage).

Parcourt un dossier `MASKS_DIR/<PatientID>/<PatientID>.nii.gz` (même arborescence que celle
lue par `seg`), isole la tumeur primaire (GTVp = label 1), mesure son diamètre maximal 3D en
mm, et en déduit le stade T :

    diamètre max ≤ 2 cm        → T1
    2 cm < diamètre max ≤ 4 cm → T2
    diamètre max > 4 cm        → T3

Écrit un CSV `PatientID, T`.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

MASKS_DIR = "dataset"                      # dossier contenant un sous-dossier par patient
OUTPUT_CSV = "tables/t_from_mask.csv"
GTVP_LABEL = 1                          # tumeur primaire dans le masque HECKTOR


def max_diameter_mm(mask: np.ndarray, affine: np.ndarray) -> float:
    """Plus grande distance (mm) entre deux voxels de la tumeur, en coordonnées physiques."""
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


rows = []
for patient_id in sorted(os.listdir(MASKS_DIR)):
    mask_path = os.path.join(MASKS_DIR, patient_id, f"{patient_id}.nii.gz")
    if not os.path.isfile(mask_path):
        continue
    img = nib.load(mask_path)
    diameter = max_diameter_mm(np.asarray(img.dataobj) == GTVP_LABEL, img.affine)
    t_stage = "T1" if diameter <= 20 else "T2" if diameter <= 40 else "T3"
    rows.append({"PatientID": patient_id, "T": t_stage})
    print(f"{patient_id}: diamètre max {diameter:.1f} mm → {t_stage}")

os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print(f"\n{len(rows)} patients → {OUTPUT_CSV}")
