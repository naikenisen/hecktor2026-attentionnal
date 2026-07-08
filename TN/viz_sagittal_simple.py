"""Visualisation minimale d'UNE seule coupe sagittale au plan médian.

Contrairement à viz_sagittal.py, ce script :
  - ne produit qu'un seul panneau (la coupe, rien d'autre) ;
  - réoriente d'abord le volume en RAS canonique (nib.as_closest_canonical)
    pour supprimer toute ambiguïté sur l'ordre des axes ;
  - prend la coupe au plan médian ANATOMIQUE (x = 0 mm en coordonnées monde),
    pas au simple milieu du tableau de voxels.

Une fois en RAS canonique, les axes sont garantis :
    axe 0 → G/D (L→R)   axe 1 → A/P (P→A)   axe 2 → I/S (I→S)
Une coupe sagittale = plan perpendiculaire à l'axe 0.

Usage :
    python -m TN.viz_sagittal_simple                     # patients de PATIENTS
    python -m TN.viz_sagittal_simple CHUM-023 MDA-436    # patients spécifiques
"""
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASKS_DIR = "dataset"
OUT_DIR   = "figures/sagittal_checks"

# Patients à visualiser. Laisser vide [] pour traiter tous les patients du dossier.
PATIENTS = [
    "CHUM-023",
]


def median_sagittal(img):
    """Retourne (slice2d, x_index, x_mm) pour la coupe sagittale au plan x = 0 mm.

    L'image est d'abord ramenée en RAS canonique : l'axe 0 pointe alors de la
    gauche vers la droite du patient. On calcule l'indice de voxel dont la
    coordonnée monde x est la plus proche de 0 (ligne médiane sagittale), puis
    on extrait le plan perpendiculaire à cet axe.

    La coupe renvoyée est orientée pour l'affichage :
        - vertical   = S/I, supérieur vers le haut
        - horizontal = A/P, antérieur vers la droite
    """
    img    = nib.as_closest_canonical(img)   # garantit l'ordre R, A, S
    data   = np.asarray(img.dataobj)
    affine = img.affine

    # Indice de voxel le plus proche de x = 0 mm sur l'axe 0 (L→R).
    # x_mm(i) = affine[0, 0] * i + affine[0, 3]
    x0     = -affine[0, 3] / affine[0, 0]
    x_idx  = int(np.clip(round(x0), 0, data.shape[0] - 1))
    x_mm   = affine[0, 0] * x_idx + affine[0, 3]

    slc = data[x_idx, :, :]      # (A/P, S/I)
    slc = slc.T                  # (S/I, A/P) : lignes = S/I, colonnes = A/P
    return slc, x_idx, float(x_mm)


def plot_patient(patient_id, path, out_dir):
    img = nib.load(path)
    slc, x_idx, x_mm = median_sagittal(img)

    fig, ax = plt.subplots(figsize=(6, 9))
    # origin="lower" → l'indice S/I 0 (inférieur) est en bas, supérieur en haut.
    ax.imshow(slc, cmap="gray", origin="lower", interpolation="nearest",
              aspect="equal")
    ax.set_title(
        f"{patient_id} — coupe sagittale médiane\n"
        f"(x = {x_mm:.1f} mm, voxel {x_idx})",
        fontsize=10,
    )
    ax.set_xlabel("A/P  (antérieur →)")
    ax.set_ylabel("S/I  (supérieur ↑)")
    ax.set_xticks([]); ax.set_yticks([])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{patient_id}_sagittal_simple.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{patient_id}: x={x_mm:.1f}mm (voxel {x_idx})  →  {out_path}")


def main():
    targets = sys.argv[1:] or PATIENTS or None

    patient_ids = sorted(os.listdir(MASKS_DIR))
    if targets:
        patient_ids = [p for p in patient_ids if p in targets]
        missing = set(targets) - set(patient_ids)
        if missing:
            print(f"Patients introuvables dans {MASKS_DIR}: {missing}")

    for patient_id in patient_ids:
        path = os.path.join(MASKS_DIR, patient_id, f"{patient_id}__PT.nii.gz")
        if not os.path.isfile(path):
            print(f"  skip {patient_id} — fichier absent")
            continue
        plot_patient(patient_id, path, OUT_DIR)

    print(f"\n{len(patient_ids)} patient(s) traité(s) → {OUT_DIR}/")


if __name__ == "__main__":
    main()
