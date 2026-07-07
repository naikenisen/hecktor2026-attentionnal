"""Visualisation de la coupe sagittale médiane + vue des deux demi-volumes.

Pour chaque patient, génère une figure PNG à 3 panneaux :
  - Panneau gauche  : coupe sagittale au plan médian (trait jaune = frontière gauche/droite)
  - Panneau centre  : demi-volume GAUCHE vu de dessus (projection axiale, MIP)
  - Panneau droite  : demi-volume DROIT vu de dessus (projection axiale, MIP)

Usage :
    python -m TN.viz_sagittal                     # tous les patients
    python -m TN.viz_sagittal MDA-436 CHUM-001    # patients spécifiques
"""
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import label as connected_components

MASKS_DIR  = "dataset"
OUT_DIR    = "figures/sagittal_checks"
SPLIT_DIR  = "figures/split_halves"        # dossier de sortie des deux demi-volumes NIfTI
GTVP_LABEL = 1
GTVN_LABEL = 2

# Patients à visualiser. Laisser vide [] pour traiter tous les patients du dossier.
PATIENTS = [
    "CHUM-023",
]

# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires axe anatomique
# ──────────────────────────────────────────────────────────────────────────────

def _axis(affine, codes):
    """Index de l'axe dont le code NIfTI est dans `codes`."""
    return next(i for i, c in enumerate(nib.aff2axcodes(affine)) if c in codes)

def _lr_axis(affine): return _axis(affine, "LR")
def _si_axis(affine): return _axis(affine, "SI")
def _ap_axis(affine):
    lr, si = _lr_axis(affine), _si_axis(affine)
    return next(i for i in range(3) if i not in (lr, si))


# ──────────────────────────────────────────────────────────────────────────────
# Logique N (copie de N.py pour rester autonome)
# ──────────────────────────────────────────────────────────────────────────────

def _tumor_side(gtvp, affine):
    if gtvp.sum() == 0:
        return None
    lr = _lr_axis(affine)
    return np.argwhere(gtvp)[:, lr].mean() < gtvp.shape[lr] / 2


def _n_stage(gtvn, affine, t_side):
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist

    def max_diam(mask):
        coords = np.argwhere(mask)
        if len(coords) < 2:
            return 0.0
        pts = nib.affines.apply_affine(affine, coords)
        if len(pts) > 3:
            try:
                pts = pts[ConvexHull(pts).vertices]
            except Exception:
                pass
        return float(pdist(pts).max())

    if gtvn.sum() == 0:
        return "N0"
    lr  = _lr_axis(affine)
    mid = gtvn.shape[lr] / 2
    comps, n = connected_components(gtvn)
    sides, max_node = set(), 0.0
    for k in range(1, n + 1):
        node = comps == k
        sides.add(np.argwhere(node)[:, lr].mean() < mid)
        max_node = max(max_node, max_diam(node))
    if max_node > 60:
        return "N3"
    if t_side is None:
        return "N2" if len(sides) > 1 else "N1"
    return "N1" if sides == {t_side} else "N2"


# ──────────────────────────────────────────────────────────────────────────────
# Séparation en deux demi-volumes
# ──────────────────────────────────────────────────────────────────────────────

def split_halves(data, affine):
    """Sépare le volume 3D en deux moitiés le long du plan médian sagittal.

    Retourne (left_vol, right_vol) : deux tableaux de même shape que `data`,
    chacun contenant uniquement les voxels de leur côté anatomique ; les
    voxels de l'autre côté sont mis à zéro.

    La détection gauche/droite utilise l'affine NIfTI (aff2axcodes) exactement
    comme N.py, ce qui garantit la cohérence avec le calcul du stade N.
    """
    lr      = _lr_axis(affine)
    lr_code = nib.aff2axcodes(affine)[lr]   # "L" ou "R"
    mid     = data.shape[lr] // 2

    idx_lo = [slice(None)] * 3
    idx_hi = [slice(None)] * 3
    idx_lo[lr] = slice(None, mid)           # indices 0 … mid-1
    idx_hi[lr] = slice(mid, None)           # indices mid … fin

    vol_lo = np.zeros_like(data)
    vol_hi = np.zeros_like(data)
    vol_lo[tuple(idx_lo)] = data[tuple(idx_lo)]
    vol_hi[tuple(idx_hi)] = data[tuple(idx_hi)]

    # Associer indices bas/haut aux côtés anatomiques
    if lr_code == "L":
        # indice croissant → direction L → lo = droite, hi = gauche
        return vol_hi, vol_lo           # (left, right)
    else:
        # indice croissant → direction R → lo = gauche, hi = droite
        return vol_lo, vol_hi           # (left, right)


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde NIfTI des deux demi-volumes
# ──────────────────────────────────────────────────────────────────────────────

def save_halves_nifti(patient_id, img, out_dir):
    """Sauvegarde deux fichiers NIfTI : côté gauche et côté droit.

    Les fichiers partagent exactement le même affine et le même header que le
    volume d'entrée — seuls les voxels de l'autre côté du plan médian sont mis
    à zéro. Ils peuvent être rouverts directement dans ITK-SNAP / 3D Slicer
    en superposition avec le volume original.

    Sorties :
        {out_dir}/{patient_id}/{patient_id}_left.nii.gz
        {out_dir}/{patient_id}/{patient_id}_right.nii.gz
    """
    data   = np.asarray(img.dataobj)
    affine = img.affine

    left_vol, right_vol = split_halves(data, affine)

    patient_dir = os.path.join(out_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    for side, vol in [("left", left_vol), ("right", right_vol)]:
        out_path = os.path.join(patient_dir, f"{patient_id}_{side}.nii.gz")
        nib.save(nib.Nifti1Image(vol, affine, img.header), out_path)
        print(f"  {side:5s} → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Projections 2D
# ──────────────────────────────────────────────────────────────────────────────

def sagittal_slice(data, affine):
    """Coupe 2D au plan médian (axe L/R), orientée SI vertical / AP horizontal."""
    lr  = _lr_axis(affine)
    mid = data.shape[lr] // 2
    slc = np.take(data, mid, axis=lr)

    codes     = list(nib.aff2axcodes(affine))
    remaining = [c for i, c in enumerate(codes) if i != lr]
    si_pos    = next((i for i, c in enumerate(remaining) if c in "SI"), 0)
    if si_pos != 0:
        slc = slc.T
    if remaining[si_pos] == "I":
        slc = slc[::-1, :]
    return slc, mid


def axial_mip(data, affine):
    """Projection axiale (vue de dessus) : max le long de l'axe S/I.

    Le résultat est orienté L→droite, A→haut pour un affichage intuitif :
    côté gauche anatomique à gauche de l'image, côté droit à droite.
    """
    si  = _si_axis(affine)
    lr  = _lr_axis(affine)
    ap  = _ap_axis(affine)

    proj = data.max(axis=si)            # (lr_dim, ap_dim) ou (ap_dim, lr_dim)

    # Assurer l'ordre (ap_dim, lr_dim) pour affichage (ap=vertical, lr=horizontal)
    axes_remaining = [a for a in range(3) if a != si]
    if axes_remaining.index(lr) == 0:  # lr est le premier axe restant → transposer
        proj = proj.T                   # maintenant (ap, lr)

    # Orienter : gauche anatomique à gauche de l'image
    lr_code = nib.aff2axcodes(affine)[lr]
    if lr_code == "L":
        proj = proj[:, ::-1]            # inverser pour que L soit à gauche

    # Orienter : antérieur en haut
    ap_code = nib.aff2axcodes(affine)[ap]
    if ap_code == "P":
        proj = proj[::-1, :]

    return proj


# ──────────────────────────────────────────────────────────────────────────────
# Superposition de couche colorée sur un axe matplotlib
# ──────────────────────────────────────────────────────────────────────────────

def _overlay(ax, img2d, color_rgb, alpha=0.55):
    rgba = np.zeros((*img2d.shape, 4))
    rgba[img2d > 0] = [*color_rgb, alpha]
    ax.imshow(rgba, origin="upper", interpolation="nearest")


def _label_comps(ax, comp_map, n):
    for k in range(1, n + 1):
        m = comp_map == k
        if m.sum() == 0:
            continue
        cy, cx = np.argwhere(m).mean(axis=0)
        ax.text(cx, cy, str(k), color="white", fontsize=8,
                ha="center", va="center", fontweight="bold")


# ──────────────────────────────────────────────────────────────────────────────
# Figure principale
# ──────────────────────────────────────────────────────────────────────────────

def plot_patient(patient_id, mask_path, out_dir):
    img    = nib.load(mask_path)
    data   = np.asarray(img.dataobj)
    affine = img.affine

    gtvp   = (data == GTVP_LABEL).astype(np.uint8)
    gtvn   = (data == GTVN_LABEL).astype(np.uint8)
    t_side = _tumor_side(gtvp, affine)
    stage  = _n_stage(gtvn, affine, t_side)
    side_str = ("gauche" if t_side else "droite") if t_side is not None else "inconnu"

    # ── Séparer en deux demi-volumes ──────────────────────────────────────────
    left_p,  right_p  = split_halves(gtvp, affine)
    left_n,  right_n  = split_halves(gtvn, affine)

    # ── Projections ───────────────────────────────────────────────────────────
    sag_bg,  mid_idx  = sagittal_slice(data, affine)
    sag_p,   _        = sagittal_slice(gtvp, affine)
    sag_n,   _        = sagittal_slice(gtvn, affine)
    sag_comps, n_sag  = connected_components(sag_n)

    ax_left_bg  = axial_mip(data,   affine)
    ax_left_p   = axial_mip(left_p,  affine)
    ax_left_n   = axial_mip(left_n,  affine)
    ax_right_p  = axial_mip(right_p, affine)
    ax_right_n  = axial_mip(right_n, affine)
    ax_left_comps,  n_left  = connected_components(ax_left_n)
    ax_right_comps, n_right = connected_components(ax_right_n)

    # ── Figure 1×3 ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    lr_code = nib.aff2axcodes(affine)[_lr_axis(affine)]

    # Panneau 1 — coupe sagittale médiane
    ax = axes[0]
    ax.imshow(sag_bg, cmap="gray", origin="upper", interpolation="nearest")
    _overlay(ax, sag_p, [1, 0, 0])
    _overlay(ax, sag_n, [0, 0.4, 1])
    _label_comps(ax, sag_comps, n_sag)
    ax.axvline(sag_bg.shape[1] / 2, color="yellow", linewidth=1.5,
               linestyle="--", label="plan médian")
    ax.set_title(
        f"Coupe sagittale médiane\n"
        f"(idx {mid_idx}, axe {lr_code})  |  GTVp côté {side_str}",
        fontsize=9,
    )
    ax.axis("off")

    # Panneau 2 — demi-volume gauche (vue axiale MIP)
    ax = axes[1]
    ax.imshow(ax_left_bg, cmap="gray", origin="upper", interpolation="nearest")
    _overlay(ax, ax_left_p, [1, 0, 0])
    _overlay(ax, ax_left_n, [0, 0.4, 1])
    _label_comps(ax, ax_left_comps, n_left)
    ax.set_title("Demi-volume GAUCHE\n(vue axiale, MIP)", fontsize=9)
    ax.axis("off")

    # Panneau 3 — demi-volume droit (vue axiale MIP)
    ax = axes[2]
    ax.imshow(ax_left_bg, cmap="gray", origin="upper", interpolation="nearest")
    _overlay(ax, ax_right_p, [1, 0, 0])
    _overlay(ax, ax_right_n, [0, 0.4, 1])
    _label_comps(ax, ax_right_comps, n_right)
    ax.set_title("Demi-volume DROIT\n(vue axiale, MIP)", fontsize=9)
    ax.axis("off")

    # Titre global + légende
    legend_elems = [
        Patch(facecolor="red",        alpha=0.55, label="GTVp (tumeur primaire)"),
        Patch(facecolor=(0, 0.4, 1),  alpha=0.55, label="GTVn (ganglions)"),
        plt.Line2D([0], [0], color="yellow", linestyle="--", label="plan médian"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.8)
    fig.suptitle(
        f"{patient_id}  —  stade N calculé : {stage}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{patient_id}_sagittal.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{patient_id}: {stage}  →  {out_path}")

    # Sauvegarde des deux demi-volumes NIfTI
    save_halves_nifti(patient_id, img, SPLIT_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────────────────────────────────────

def main():
    targets = sys.argv[1:] or PATIENTS or None

    patient_ids = sorted(os.listdir(MASKS_DIR))
    if targets:
        patient_ids = [p for p in patient_ids if p in targets]
        missing = set(targets) - set(patient_ids)
        if missing:
            print(f"Patients introuvables dans {MASKS_DIR}: {missing}")

    for patient_id in patient_ids:
        mask_path = os.path.join(MASKS_DIR, patient_id, f"{patient_id}__PT.nii.gz")
        if not os.path.isfile(mask_path):
            print(f"  skip {patient_id} — masque absent")
            continue
        plot_patient(patient_id, mask_path, OUT_DIR)

    print(f"\n{len(patient_ids)} patient(s) traité(s) → {OUT_DIR}/")


if __name__ == "__main__":
    main()
