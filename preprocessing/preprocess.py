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
from convert_nifti_Bq_SUV import bq_to_suv

def sitk_to_metatensor(img_sitk: sitk.Image) -> MetaTensor:
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    arr = arr[None, ...]
    spacing   = np.array(img_sitk.GetSpacing())
    origin    = np.array(img_sitk.GetOrigin())
    direction = np.array(img_sitk.GetDirection()).reshape(3, 3)
    A_lps = direction * spacing
    lps_to_ras = np.array([-1.0, -1.0, 1.0])
    A_ras      = A_lps * lps_to_ras[:, np.newaxis]
    origin_ras = origin * lps_to_ras
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

def get_bounding_boxes(ct_sitk, pet_sitk):
    ct_origin = np.array(ct_sitk.GetOrigin())
    pet_origin = np.array(pet_sitk.GetOrigin())
    ct_position_max = ct_origin + np.array(ct_sitk.GetSize()) * np.array(ct_sitk.GetSpacing())
    pet_position_max = pet_origin + np.array(pet_sitk.GetSize()) * np.array(pet_sitk.GetSpacing())
    return np.concatenate([
        np.maximum(ct_origin, pet_origin),
        np.minimum(ct_position_max, pet_position_max),
    ], axis=0)

def resample_images(ct_path, pet_path, lbl_path=None):
    resampling = [1, 1, 1]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)
    ct = sitk.ReadImage(ct_path)
    pt = sitk.ReadImage(pet_path)
    bb = get_bounding_boxes(ct, pt)
    size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])
    resampler.SetInterpolator(sitk.sitkBSpline)
    ct = resampler.Execute(ct)
    pt = resampler.Execute(pt)
    lbl = None
    if lbl_path is not None:
        lbl = sitk.ReadImage(lbl_path)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        lbl = resampler.Execute(lbl)
    return ct, pt, bb, lbl

def get_roi_center(pet_tensor, z_top_fraction=0.75, z_score_threshold=1.0):
    image_shape_voxels = np.array(pet_tensor.shape)
    crop_z_start = int(z_top_fraction * image_shape_voxels[2])
    top_of_scan = pet_tensor[..., crop_z_start:]
    mask = ((top_of_scan - top_of_scan.mean()) / (top_of_scan.std() + 1e-8)) > z_score_threshold
    if not mask.any():
        warnings.warn("No high-intensity region found. Using geometric center of the upper scan region.")
        center_in_top = (np.array(top_of_scan.shape) / 2).astype(int)
    else:
        labeled_mask, num_features = label(mask, return_num=True, connectivity=3)
        if num_features > 0:
            component_sizes = np.bincount(labeled_mask.ravel())[1:]
            largest_component_label = np.argmax(component_sizes) + 1
            largest_component_mask = labeled_mask == largest_component_label
            comp_idx = np.argwhere(largest_component_mask)
        else:
            comp_idx = np.argwhere(mask)
        center_in_top = np.mean(comp_idx, axis=0)
    center_full_image = center_in_top + np.array([0, 0, crop_z_start])
    return center_full_image.astype(int)

def crop_neck_region_sitk(
        ct_sitk:  sitk.Image,
        pet_sitk: sitk.Image,
        lbl_sitk: sitk.Image = None,
        crop_box_size=(200, 200, 310),
        z_top_fraction=0.75,
        z_score_threshold=1.0,
):
    pet_np_zyx = sitk.GetArrayFromImage(pet_sitk)
    pet_np_xyz = np.transpose(pet_np_zyx, (2, 1, 0))
    pet_tensor = torch.from_numpy(pet_np_xyz).float()
    crop_box_size = np.asarray(crop_box_size, dtype=int)
    center = get_roi_center(pet_tensor,
                            z_top_fraction=z_top_fraction,
                            z_score_threshold=z_score_threshold)
    img_shape = np.asarray(pet_np_xyz.shape)
    box_start = np.clip(center - crop_box_size // 2, 0, img_shape)
    box_end = np.clip(box_start + crop_box_size, 0, img_shape)
    box_start = np.maximum(box_end - crop_box_size, 0)
    index = [int(i) for i in box_start]
    size = [int(e - s) for s, e in zip(box_start, box_end)]
    ct_crop = sitk.RegionOfInterest(ct_sitk, size=size, index=index)
    pet_crop = sitk.RegionOfInterest(pet_sitk, size=size, index=index)
    lbl_crop = sitk.RegionOfInterest(lbl_sitk, size=size, index=index) if lbl_sitk is not None else None
    return ct_crop, pet_crop, lbl_crop, box_start, box_end

def get_preprocessing_transforms(keys, final_size=(200, 200, 310)):
    return Compose([
        Orientationd(keys=keys, axcodes="RAS", labels=None),
        ScaleIntensityRanged(
            keys=["ct"], a_min=-250, a_max=250, b_min=-6.0, b_max=6.0, clip=True
        ),
        NormalizeIntensityd(keys=["pet"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key="ct", allow_smaller=True),
        SpatialPadd(keys=keys, spatial_size=final_size, method="end"),
    ])

def apply_monai_transforms(ct_sitk: sitk.Image,
                            pt_sitk: sitk.Image,
                            lbl_sitk: sitk.Image = None,
                            final_size=(200, 200, 310)):
    ct_mt = sitk_to_metatensor(ct_sitk)
    pet_mt = sitk_to_metatensor(pt_sitk)
    data = {"ct": ct_mt, "pet": pet_mt}
    keys = ["ct", "pet"]
    if lbl_sitk is not None:
        data["label"] = sitk_to_metatensor(lbl_sitk)
        keys = ["ct", "pet", "label"]
    xforms = get_preprocessing_transforms(keys=keys, final_size=final_size)
    out = xforms(data)
    ct_proc = out["ct"]
    pet_proc = out["pet"]
    meta = ct_proc.meta
    lbl_proc = out["label"] if lbl_sitk is not None else None
    return ct_proc, pet_proc, lbl_proc, meta

def _find_modality_file(pdir: Path, pid: str, modality: str) -> Path | None:
    """Find CT or PT nifti file trying multiple naming conventions."""
    patterns = [
        f"{pid}__{modality}.nii.gz",
        f"{pid}_{modality}.nii.gz",
        f"{pid}__{modality.lower()}.nii.gz",
        f"{pid}_{modality.lower()}.nii.gz",
    ]
    for pat in patterns:
        p = pdir / pat
        if p.exists():
            return p
    # Fall back: any nifti in the dir containing the modality tag
    for p in pdir.glob("*.nii.gz"):
        stem = p.name.replace(".nii.gz", "").upper()
        if modality.upper() in stem.split("_") or stem.endswith(f"_{modality.upper()}") or stem.endswith(f"__{modality.upper()}"):
            return p
    return None

def _find_label_file(pdir: Path, pid: str) -> Path | None:
    for name in [f"{pid}.nii.gz", f"{pid}_label.nii.gz", f"{pid}__label.nii.gz", f"{pid}_seg.nii.gz"]:
        p = pdir / name
        if p.exists():
            return p
    return None

def process_patient(args):
    """Traite un patient : rééchantillonnage, recadrage, preprocessing MONAI et sauvegarde."""
    pdir, output_dir = args
    pid      = pdir.name
    ct_path  = _find_modality_file(pdir, pid, "CT")
    pt_path  = _find_modality_file(pdir, pid, "PT")
    lbl_path = _find_label_file(pdir, pid)
    if ct_path is None or pt_path is None:
        return pid, "missing"
    try:
        out_pdir = output_dir / pid
        out_pdir.mkdir(parents=True, exist_ok=True)
        has_label = lbl_path is not None
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
    input_dir  = Path("/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/dataset/HECKTOR 2026 Training Data")
    output_dir = Path("/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/dataset/hecktor_dataset_suv")
    max_workers = max(1, multiprocessing.cpu_count() // 2)
    n_workers   = max_workers

    #Bq to SUV conversion (if needed)
    bq_to_suv(
        input_root=str(input_dir),
        headers_csv= "/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/preprocessing/suv_conversion_tags.csv",
        output_root=str(input_dir),)
    #End of Bq to SUV conversion
    
    patient_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    print(f"{len(patient_dirs)} patients trouves dans {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    errors   = []
    skipped  = 0
    done_pids: set = set()
    all_args = [(pdir, output_dir) for pdir in patient_dirs]
    remaining = list(all_args)
    pbar = tqdm(total=len(all_args), desc="Preprocessing")
    while remaining:
        print(f"Demarrage avec {n_workers} workers ({len(remaining)} patients restants)")
        batch = list(remaining)
        remaining = []
        batch_ok = True
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_patient, a): a[0].name for a in batch}
                for future in concurrent.futures.as_completed(futures):
                    pid_key = futures[future]
                    try:
                        pid, result = future.result()
                    except concurrent.futures.process.BrokenProcessPool:
                        raise
                    except Exception as exc:
                        done_pids.add(pid_key)
                        pbar.update(1)
                        errors.append((pid_key, exc))
                        print(f"\n  [{pid_key}] ERREUR inattendue : {exc}")
                        continue
                    done_pids.add(pid)
                    pbar.update(1)
                    if result == "missing":
                        skipped += 1
                        print(f"\n  [{pid}] CT ou PT manquant, ignore")
                    elif result is not None:
                        errors.append((pid, result))
                        print(f"\n  [{pid}] ERREUR : {result}")
        except concurrent.futures.process.BrokenProcessPool:
            batch_ok = False
            remaining = [a for a in batch if a[0].name not in done_pids]
            n_workers = max(1, n_workers // 2)
            print(f"\nPool casse (OOM kernel). {len(done_pids)} patients traites. "
                  f"Reprise avec {n_workers} worker(s) pour {len(remaining)} patients restants.")
        if batch_ok and n_workers < max_workers:
            n_workers = min(n_workers * 2, max_workers)
            print(f"\nBatch reussi, remontee a {n_workers} workers.")
    pbar.close()
    processed = len(all_args) - len(errors) - skipped
    print(f"{processed}/{len(all_args)}")
    print(f"{output_dir}")
    if errors:
        print(f"{len(errors)} erreur(s) :")
        for pid, exc in errors:
            print(f"  {pid}: {exc}")


if __name__ == "__main__":
    main()
