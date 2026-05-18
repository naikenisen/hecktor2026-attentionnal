import concurrent.futures
import multiprocessing
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import torch
import warnings
from pathlib import Path
from tqdm import tqdm
from skimage.measure import label

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    NormalizeIntensityd,
    Activationsd,
)
from monai.data import MetaTensor


# Convertit une image SimpleITK en MetaTensor MONAI channel-first [1, Z, Y, X]
def sitk_to_metatensor(img_sitk: sitk.Image) -> MetaTensor:
    # Tableau numpy [Z, Y, X] extrait de l'image SimpleITK
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    # Ajout de la dimension canal : [1, Z, Y, X]
    arr = arr[None, ...]

    spacing   = np.array(img_sitk.GetSpacing())                    # (sx, sy, sz) en LPS
    origin    = np.array(img_sitk.GetOrigin())                     # (ox, oy, oz) en LPS
    direction = np.array(img_sitk.GetDirection()).reshape(3, 3)    # 3×3 cosinus directeurs LPS

    # Affine LPS pour des indices voxel en ordre (x, y, z)
    A_lps = direction * spacing  # colonne j = direction[:,j] * spacing[j]

    # Conversion LPS → RAS : inverser les lignes x et y (et l'origine correspondante)
    # SimpleITK suit la convention DICOM (LPS) ; nibabel/MONAI utilisent RAS
    lps_to_ras = np.array([-1.0, -1.0, 1.0])
    A_ras      = A_lps * lps_to_ras[:, np.newaxis]
    origin_ras = origin * lps_to_ras

    # Inverser les colonnes pour tenir compte de l'ordre (Z, Y, X) du tableau numpy
    # (sitk.GetArrayFromImage retourne [z, y, x] mais l'affine LPS suppose [x, y, z])
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = A_ras[:, ::-1]
    affine[:3, 3]  = origin_ras

    meta = {
        "spacing":   tuple(img_sitk.GetSpacing()),
        "origin":    tuple(img_sitk.GetOrigin()),
        "direction": tuple(img_sitk.GetDirection()),
        "affine":    affine,
        "space":     "RAS",
    }
    return MetaTensor(arr, meta=meta)


# Calcule l'intersection des bounding boxes CT et PET en coordonnées physiques
def get_bounding_boxes(ct_sitk, pet_sitk):
    # Origine physique du CT
    ct_origin = np.array(ct_sitk.GetOrigin())
    # Origine physique du PET
    pet_origin = np.array(pet_sitk.GetOrigin())

    # Coin supérieur du CT (origin + taille × spacing)
    ct_position_max = ct_origin + np.array(ct_sitk.GetSize()) * np.array(ct_sitk.GetSpacing())
    # Coin supérieur du PET
    pet_position_max = pet_origin + np.array(pet_sitk.GetSize()) * np.array(pet_sitk.GetSpacing())

    # Retourne [max_origin | min_max] : la bounding box commune CT/PET
    return np.concatenate([
        np.maximum(ct_origin, pet_origin),
        np.minimum(ct_position_max, pet_position_max),
    ], axis=0)


# Rééchantillonne CT, PET et optionnellement le label à 1 mm³ isotrope dans leur bounding box commune
def resample_images(ct_path, pet_path, lbl_path=None):
    # Résolution cible isotrope en mm
    resampling = [1, 1, 1]
    # Filtre de rééchantillonnage SimpleITK configuré une seule fois
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)
    # Lecture du volume CT
    ct = sitk.ReadImage(ct_path)
    # Lecture du volume PET
    pt = sitk.ReadImage(pet_path)
    # Bounding box commune en coordonnées physiques
    bb = get_bounding_boxes(ct, pt)
    # Taille en voxels de la bounding box commune
    size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])
    resampler.SetInterpolator(sitk.sitkBSpline)
    # CT rééchantillonné
    ct = resampler.Execute(ct)
    # PET rééchantillonné
    pt = resampler.Execute(pt)

    lbl = None
    if lbl_path is not None:
        lbl = sitk.ReadImage(lbl_path)
        # NearestNeighbor pour préserver les valeurs entières du masque de segmentation
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        lbl = resampler.Execute(lbl)

    return ct, pt, bb, lbl


# Trouve le centre de la région haute intensité dans la partie supérieure du PET
def get_roi_center(pet_tensor, z_top_fraction=0.75, z_score_threshold=1.0):
    # Forme du volume en voxels
    image_shape_voxels = np.array(pet_tensor.shape)
    # Slice de début pour la partie supérieure du scan (tête/cou)
    crop_z_start = int(z_top_fraction * image_shape_voxels[2])
    # Sous-volume correspondant au haut du scan
    top_of_scan = pet_tensor[..., crop_z_start:]

    # Masque binaire des voxels au-dessus du seuil de z-score
    mask = ((top_of_scan - top_of_scan.mean()) / (top_of_scan.std() + 1e-8)) > z_score_threshold

    if not mask.any():
        warnings.warn("No high-intensity region found. Using geometric center of the upper scan region.")
        # Centroïde géométrique du sous-volume si aucun voxel n'est au-dessus du seuil
        center_in_top = (np.array(top_of_scan.shape) / 2).astype(int)
    else:
        # Étiquetage des composantes connexes dans le masque
        labeled_mask, num_features = label(mask, return_num=True, connectivity=3)
        if num_features > 0:
            # Tailles des composantes connexes (hors fond)
            component_sizes = np.bincount(labeled_mask.ravel())[1:]
            # Label de la plus grande composante connexe
            largest_component_label = np.argmax(component_sizes) + 1
            # Masque de la plus grande composante
            largest_component_mask = labeled_mask == largest_component_label
            # Indices voxels de la composante sélectionnée
            comp_idx = np.argwhere(largest_component_mask)
        else:
            comp_idx = np.argwhere(mask)

        # Centroïde de la composante sélectionnée dans le sous-volume
        center_in_top = np.mean(comp_idx, axis=0)

    # Centre ramené dans le repère du volume complet
    center_full_image = center_in_top + np.array([0, 0, crop_z_start])
    return center_full_image.astype(int)


# Recadre CT, PET et optionnellement le label autour de la région cou/tête via une boîte centrée sur le ROI PET
def crop_neck_region_sitk(
        ct_sitk:  sitk.Image,
        pet_sitk: sitk.Image,
        lbl_sitk: sitk.Image = None,
        crop_box_size=(200, 200, 310),
        z_top_fraction=0.75,
        z_score_threshold=1.0,
):
    # Conversion PET en numpy [z,y,x] puis transposé en [x,y,z] pour le calcul du centre
    pet_np_zyx = sitk.GetArrayFromImage(pet_sitk)
    pet_np_xyz = np.transpose(pet_np_zyx, (2, 1, 0))
    # Tensor PyTorch du PET pour get_roi_center
    pet_tensor = torch.from_numpy(pet_np_xyz).float()

    # Taille de la boîte de recadrage en voxels
    crop_box_size = np.asarray(crop_box_size, dtype=int)
    # Centre du ROI estimé depuis l'intensité PET
    center = get_roi_center(pet_tensor,
                            z_top_fraction=z_top_fraction,
                            z_score_threshold=z_score_threshold)

    # Forme du volume en voxels
    img_shape = np.asarray(pet_np_xyz.shape)
    # Coin inférieur de la boîte (clampé dans le volume)
    box_start = np.clip(center - crop_box_size // 2, 0, img_shape)
    # Coin supérieur de la boîte (clampé dans le volume)
    box_end = np.clip(box_start + crop_box_size, 0, img_shape)
    # Correction du coin inférieur si l'image est plus petite que la boîte
    box_start = np.maximum(box_end - crop_box_size, 0)

    # Index et taille du crop en ordre (x, y, z) pour SimpleITK
    index = [int(i) for i in box_start]
    size = [int(e - s) for s, e in zip(box_start, box_end)]

    # CT recadré autour de la région cou/tête
    ct_crop = sitk.RegionOfInterest(ct_sitk, size=size, index=index)
    # PET recadré de façon identique
    pet_crop = sitk.RegionOfInterest(pet_sitk, size=size, index=index)
    # Label recadré avec le même index/size pour rester aligné sur CT/PET
    lbl_crop = sitk.RegionOfInterest(lbl_sitk, size=size, index=index) if lbl_sitk is not None else None

    return ct_crop, pet_crop, lbl_crop, box_start, box_end


# Construit la séquence de transforms MONAI pour le preprocessing déterministe
def get_preprocessing_transforms(keys, final_size=(200, 200, 310)):
    return Compose([
        # Réorientation vers le repère RAS standard
        Orientationd(keys=keys, axcodes="RAS", labels=None),
        # Mise à l'échelle CT dans [−6, 6] en clampant à [−250, 250] HU
        ScaleIntensityRanged(
            keys=["ct"], a_min=-250, a_max=250, b_min=-6.0, b_max=6.0, clip=True
        ),
        # Normalisation PET à moyenne nulle et variance unité (sur les voxels non nuls)
        NormalizeIntensityd(keys=["pet"], nonzero=True, channel_wise=True),
        # Recadre le fond (air) basé sur le CT puis pad à la taille cible
        CropForegroundd(keys=keys, source_key="ct", allow_smaller=True),
        # Padding spatial pour garantir une taille uniforme entre patients
        SpatialPadd(keys=keys, spatial_size=final_size, method="end"),
    ])


# Applique le pipeline MONAI de preprocessing sur des volumes SimpleITK déjà en mémoire
def apply_monai_transforms(ct_sitk: sitk.Image,
                            pt_sitk: sitk.Image,
                            lbl_sitk: sitk.Image = None,
                            final_size=(200, 200, 310)):
    # Conversion CT SimpleITK → MetaTensor MONAI
    ct_mt = sitk_to_metatensor(ct_sitk)
    # Conversion PET SimpleITK → MetaTensor MONAI
    pet_mt = sitk_to_metatensor(pt_sitk)
    # Dictionnaire d'entrée pour le Compose MONAI
    data = {"ct": ct_mt, "pet": pet_mt}
    keys = ["ct", "pet"]

    if lbl_sitk is not None:
        data["label"] = sitk_to_metatensor(lbl_sitk)
        keys = ["ct", "pet", "label"]

    # Pipeline de preprocessing déterministe (Orientationd, CropForegroundd, SpatialPadd appliqués à toutes les keys)
    xforms = get_preprocessing_transforms(keys=keys, final_size=final_size)
    # Application des transforms
    out = xforms(data)

    # CT preprocessé : MetaTensor 1×H×W×D
    ct_proc = out["ct"]
    # PET preprocessé : MetaTensor 1×H×W×D
    pet_proc = out["pet"]
    # Métadonnées spatiales du CT après preprocessing
    meta = ct_proc.meta
    lbl_proc = out["label"] if lbl_sitk is not None else None

    return ct_proc, pet_proc, lbl_proc, meta


def process_patient(args):
    """Traite un patient : rééchantillonnage, recadrage, preprocessing MONAI et sauvegarde."""
    pdir, output_dir = args
    pid      = pdir.name
    ct_path  = pdir / f"{pid}__CT.nii.gz"
    pt_path  = pdir / f"{pid}__PT.nii.gz"
    lbl_path = pdir / f"{pid}.nii.gz"

    if not ct_path.exists() or not pt_path.exists():
        return pid, "missing"

    try:
        out_pdir = output_dir / pid
        out_pdir.mkdir(parents=True, exist_ok=True)

        has_label = lbl_path.exists()
        ct_sitk, pt_sitk, _, lbl_sitk = resample_images(
            str(ct_path), str(pt_path),
            str(lbl_path) if has_label else None,
        )
        ct_crop, pt_crop, lbl_crop, _, _ = crop_neck_region_sitk(ct_sitk, pt_sitk, lbl_sitk)
        ct_proc, pet_proc, lbl_proc, _   = apply_monai_transforms(ct_crop, pt_crop, lbl_crop)

        ct_affine  = np.array(ct_proc.meta.get("affine",  np.eye(4)))
        pet_affine = np.array(pet_proc.meta.get("affine", np.eye(4)))

        nib.save(
            nib.Nifti1Image(ct_proc.numpy().squeeze(0), ct_affine),
            str(out_pdir / f"{pid}__CT.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(pet_proc.numpy().squeeze(0), pet_affine),
            str(out_pdir / f"{pid}__PT.nii.gz"),
        )

        if has_label and lbl_proc is not None:
            lbl_affine = np.array(lbl_proc.meta.get("affine", np.eye(4)))
            nib.save(
                nib.Nifti1Image(lbl_proc.numpy().squeeze(0).astype(np.uint8), lbl_affine),
                str(out_pdir / f"{pid}.nii.gz"),
            )

        return pid, None

    except Exception as exc:
        return pid, exc


def main() -> None:
    input_dir  = Path("/work/imvia/in156281/datasets/hecktor_dataset")
    output_dir = Path("/work/imvia/in156281/datasets/hecktor_dataset_preprocessed")
    n_workers  = max(1, multiprocessing.cpu_count() // 2)

    patient_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    print(f"{len(patient_dirs)} patients trouvés dans {input_dir}")
    print(f"Parallélisation sur {n_workers} workers CPU")
    output_dir.mkdir(parents=True, exist_ok=True)

    errors  = []
    skipped = 0
    args    = [(pdir, output_dir) for pdir in patient_dirs]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_patient, a): a[0].name for a in args}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Preprocessing",
        ):
            pid, result = future.result()
            if result == "missing":
                skipped += 1
                print(f"  [{pid}] CT ou PT manquant, ignoré")
            elif result is not None:
                errors.append((pid, result))
                print(f"  [{pid}] ERREUR : {result}")

    processed = len(patient_dirs) - len(errors) - skipped
    print(f"{processed}/{len(patient_dirs)}")
    print(f"{output_dir}")
    if errors:
        print(f"{len(errors)} erreur(s) :")
        for pid, exc in errors:
            print(f"  {pid}: {exc}")


if __name__ == "__main__":
    main()
