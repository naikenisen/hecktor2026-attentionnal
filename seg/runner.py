"""Pilotage du `nnUNetV2Runner` (MONAI) : instanciation, planification + prétraitement,
entraînement d'un fold, et lecture du Dice de validation produit par nnU-Net.
"""
import os
import json
from . import config
from monai.apps.nnunet import nnUNetV2Runner


def build_runner(datalist_path: str) -> nnUNetV2Runner:
    """Instancie le runner. La conversion (`convert_dataset`) n'est pas utilisée — on alimente
    nnU-Net directement (cf. `seg.dataset`) — mais `datalist`/`dataroot`/`modality` restent
    requis par l'`input_config`."""
    return nnUNetV2Runner(
        input_config={
            "dataset_name_or_id": config.nnunet_dataset_id,
            "datalist": datalist_path,
            "dataroot": config.data_root,
            "modality": config.nnunet_modality,
            "nnunet_raw": config.nnunet_raw_dir,
            "nnunet_preprocessed": config.nnunet_preprocessed_dir,
            "nnunet_results": config.nnunet_results_dir,
        },
        trainer_class_name=config.nnunet_trainer,
        work_dir=config.results_dir,
    )


def plan_and_process(runner: nnUNetV2Runner):
    """Empreinte + planification + prétraitement de la seule configuration entraînée."""
    runner.plan_and_process(
        verify_dataset_integrity=True,
        gpu_memory_target=config.nnunet_gpu_memory_gb,
        c=(config.nnunet_configuration,),
        n_proc=(config.num_workers,),
        npfp=config.num_workers,
    )


def train_fold(runner: nnUNetV2Runner):
    """Entraîne la configuration et le fold demandés (3d_fullres / fold 0 par défaut)."""
    runner.train_single_model(
        config=config.nnunet_configuration,
        fold=config.nnunet_fold,
        gpu_id=0,
    )


def report_validation():
    """Affiche le Dice de validation (moyenne foreground + par classe) écrit par nnU-Net."""
    summary = os.path.join(
        config.nnunet_model_dir, f"fold_{config.nnunet_fold}", "validation", "summary.json")
    if not os.path.exists(summary):
        print(f"[nnunet] summary de validation introuvable ({summary})")
        return
    with open(summary) as f:
        data = json.load(f)
    mean = data.get("foreground_mean", {}).get("Dice")
    per_class = {k: v.get("Dice") for k, v in data.get("mean", {}).items()}
    print(f"[nnunet] Dice validation (moyenne foreground) {mean}")
    print(f"[nnunet] Dice par classe (label → Dice) {per_class}")
