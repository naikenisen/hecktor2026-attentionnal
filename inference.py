"""
HECKTOR 2026 Challenge — Inference Entry Point

This script is executed at container runtime on Grand Challenge.
It reads a paired CT + PET scan and clinical data (EHR) from /input,
runs the three-stage pipeline, and writes all outputs to /output:

  Subtask 1 — Segmentation:
      /output/images/head-neck-tumor-segmentation/output.mha
        (label 0=background, 1=GTVp, 2=GTVn)

  Subtask 2 — TN Staging:
      /output/t-stage.json   → "T2"
      /output/n-stage.json   → "N1"

  Subtask 3 — Prognosis (RFS):
      /output/rfs.json   → <float> RFS time

Model weights are loaded from /opt/ml/model at runtime (uploaded separately
as a tarball to Grand Challenge via Algorithm > Models).

To test locally:
    ./do_test_run.sh

To save and upload:
    ./do_save.sh
"""

import json
import os
import tempfile
from glob import glob
from pathlib import Path
import pandas as pd
import SimpleITK
import numpy as np

from TN.T import t_from_mask
from TN.N import n_stage, tumor_side, node_features
import joblib

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")


def run():
    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    ct_path = get_image_file(INPUT_PATH / "images/ct")
    pet_path = get_image_file(INPUT_PATH / "images/pet")
    ehr = load_json(INPUT_PATH / "ehr.json")

    # ------------------------------------------------------------------
    # 2. Subtask 1 — Segmentation
    #    Replace the block below with your model's inference logic.
    #    Load your weights from RESOURCE_PATH / "checkpoints".
    # ------------------------------------------------------------------
    segmentation_array = run_segmentation(ct_path, pet_path)

    write_segmentation(
        location=OUTPUT_PATH / "images/head-neck-tumor-segmentation",
        array=segmentation_array,
        reference_path=ct_path,
    )

    # ------------------------------------------------------------------
    # 3. Subtask 2 — TN Staging
    #    Replace the block below with your model's inference logic.
    #    You may use segmentation_array as an additional input.
    # ------------------------------------------------------------------
    t_stage, n_stage = run_tn_staging(ct_path, segmentation_array)

    write_json(location=OUTPUT_PATH / "t-stage.json", data=t_stage)
    write_json(location=OUTPUT_PATH / "n-stage.json", data=n_stage)

    # ------------------------------------------------------------------
    # 4. Subtask 3 — Prognosis
    #    Replace the block below with your model's inference logic.
    #    You may use segmentation_array and tn stage as additional inputs.
    # ------------------------------------------------------------------
    rfs_score = run_prognosis(ehr)

    write_json(location=OUTPUT_PATH / "rfs.json", data=float(rfs_score))

    return 0


# =============================================================================
# Subtask implementations — replace with your own model code
# =============================================================================

def _sitk_affine_zyx(sitk_image):
    """4×4 affine mappant les indices numpy ZYX (depuis SimpleITK) vers les coordonnées monde en mm.

    SimpleITK stocke ses axes en ordre (i, j, k) et les tableaux numpy sont en ordre ZYX
    (k=axis0, j=axis1, i=axis2). On permute les colonnes de la direction pour que l'affine
    mappe correctement (k_idx, j_idx, i_idx) → coordonnées physiques.
    """
    d = np.array(sitk_image.GetDirection()).reshape(3, 3)  # col[0]=dir_i, col[1]=dir_j, col[2]=dir_k
    s = np.array(sitk_image.GetSpacing())                  # (si, sj, sk)
    o = np.array(sitk_image.GetOrigin())
    affine = np.eye(4, dtype=float)
    affine[:3, 0] = d[:, 2] * s[2]   # axe numpy 0 → axe ITK k (crânio-caudal)
    affine[:3, 1] = d[:, 1] * s[1]   # axe numpy 1 → axe ITK j (antéro-postérieur)
    affine[:3, 2] = d[:, 0] * s[0]   # axe numpy 2 → axe ITK i (gauche-droite)
    affine[:3, 3] = o
    return affine


def run_segmentation(ct_path, pet_path):
    """
    Charge le modèle nnUNet depuis MODEL_PATH et prédit la segmentation.
    Retourne un tableau numpy ZYX avec les labels : 0=background, 1=GTVp, 2=GTVn.
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_folder = str(
        MODEL_PATH / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_truncated = os.path.join(tmp_dir, "case")
        predictor.predict_from_files(
            [[str(ct_path), str(pet_path)]],
            [out_truncated],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        seg_img = SimpleITK.ReadImage(out_truncated + ".nii.gz")
        return SimpleITK.GetArrayFromImage(seg_img).astype(np.uint8)


def run_tn_staging(ct_path, segmentation_array):
    """
    Load your TN staging model from MODEL_PATH and run inference.
    Returns (t_stage: str, n_stage: str), e.g. ("T2", "N1").
    TN staging follows AJCC/UICC 7th Edition (N2b and N2c collapsed to N2).
    """
    ct_image = SimpleITK.ReadImage(ct_path)
    affine = _sitk_affine_zyx(ct_image)
    spacing = ct_image.GetSpacing()                          # (sx, sy, sz) — seul min() est utilisé

    gtvp = (segmentation_array == 1).astype(np.uint8)
    gtvn = (segmentation_array == 2).astype(np.uint8)

    t = t_from_mask(segmentation_array, affine, spacing)
    model_rf_nodes = joblib.load(MODEL_PATH / "n-staging.joblib")
    n_nodes, largest_node_mm, nodes_ipsilateral = node_features(gtvn, affine, tumor_side(gtvp, affine))
    features_n = pd.DataFrame([{
        "n_nodes": n_nodes,
        "largest_node_mm": largest_node_mm,
        "nodes_ipsilateral": nodes_ipsilateral,
    }])
    n = model_rf_nodes.predict(features_n)[0]
    # n = n_stage(gtvn, affine, tumor_side(gtvp, affine))
    return t, n


def run_prognosis(ehr):
    """
    Load your prognosis model from MODEL_PATH and run inference.
    Returns a float RFS Time. The output should be anti-concordant with the predicted risk score (i.e., the model should output RFS in days)
    """
    model = joblib.load(MODEL_PATH / "rfs.joblib")
    features = pd.DataFrame([{
        "Age": ehr["Age"],
        "Gender": ehr["Gender"],
        "Tobacco Consumption": ehr["Tobacco Consumption"],
        "Alcohol Consumption": ehr["Alcohol Consumption"],
        "Performance Status": ehr["Performance Status"],
        "HPV Status": ehr["HPV Status"],
        "Treatment": ehr["Treatment"],
    }])
    risk_score = model.predict(features)[0]

    # RandomSurvivalForest.predict returns a risk score (higher = shorter RFS),
    # so it must be negated to be anti-concordant with the predicted risk.
    return -1.0 * risk_score


# =============================================================================
# I/O utilities
# =============================================================================

def get_image_file(location):
    files = (
        glob(str(location / "*.mha"))
        + glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.nii.gz"))
    )
    if not files:
        raise FileNotFoundError(f"No image file found in {location}")
    return files[0]


def load_json(location):
    with open(location, "r") as f:
        return json.load(f)


def write_json(location, data):
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        json.dump(data, f, indent=2)


def write_segmentation(location, array, reference_path):
    location.mkdir(parents=True, exist_ok=True)
    reference = SimpleITK.ReadImage(reference_path)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    img = SimpleITK.GetImageFromArray(array.astype(np.uint8))
    img.CopyInformation(reference)
    SimpleITK.WriteImage(img, str(location / "output.mha"), useCompression=True)


if __name__ == "__main__":
    raise SystemExit(run())