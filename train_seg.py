"""Phase 1 — Segmentation nnU-Net (MONAI `nnUNetV2Runner` pilotant le paquet `nnunetv2`).

Remplace l'ancien backbone SwinUNETR + recherche d'hyperparamètres Optuna. nnU-Net est
auto-configurant : il dérive patch size, batch size, normalisation et data augmentation
de l'empreinte (« fingerprint ») du jeu de données. Il n'y a donc plus ni recherche HP ni
retrain séparé — l'entraînement de cette phase est complet à lui seul.

Pipeline :
  1. Construit l'arborescence brute nnU-Net (`imagesTr`/`labelsTr` + `dataset.json`) à partir
     des volumes CT/PET/masque déjà prétraités, via des liens symboliques (CT → canal 0000,
     PET → canal 0001). On contourne `runner.convert_dataset()` qui exige un unique volume
     multi-canaux par cas et renomme les patients en `case_N`.
  2. Empreinte + planification + prétraitement (`plan_and_process`).
  3. Fige le split train/val déterministe du projet (`split_case_ids`) en fold 0
     (`splits_final.json`), pour que la validation porte sur exactement les mêmes patients
     que le reste de la pipeline.
  4. Entraîne la configuration demandée (3d_fullres par défaut) sur ce fold.
"""
import os
import json
import config

# nnU-Net lit ces variables d'environnement dès l'import de `nnunetv2` : les fixer avant.
os.environ["nnUNet_raw"] = config.nnunet_raw_dir
os.environ["nnUNet_preprocessed"] = config.nnunet_preprocessed_dir
os.environ["nnUNet_results"] = config.nnunet_results_dir

import pandas as pd
from monai.apps.nnunet import nnUNetV2Runner
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from src.clinical_data import split_case_ids


def build_records(case_ids: list, known_patients: set) -> list:
    """(case_id, ct, pet, label) pour les patients du CSV dont les 3 volumes existent."""
    records = []
    for case_id in case_ids:
        if case_id not in known_patients:
            continue
        patient_dir = os.path.join(config.data_root, case_id)
        ct = os.path.join(patient_dir, f"{case_id}__CT.nii.gz")
        pet = os.path.join(patient_dir, f"{case_id}__PT.nii.gz")
        label = os.path.join(patient_dir, f"{case_id}.nii.gz")
        if not (os.path.exists(ct) and os.path.exists(pet) and os.path.exists(label)):
            continue
        records.append((case_id, ct, pet, label))
    return records


def _symlink(src: str, dst: str):
    if os.path.lexists(dst):
        return
    os.symlink(os.path.abspath(src), dst)


def prepare_raw_dataset(records: list) -> str:
    """Crée `Dataset<ID>_<name>/{imagesTr,labelsTr}` (liens symboliques) + `dataset.json`,
    et un datalist MONAI (trace + clé requise par l'`input_config` du runner). Retourne le
    chemin du datalist."""
    raw_dir = os.path.join(config.nnunet_raw_dir, config.nnunet_dataset_dir)
    images_dir = os.path.join(raw_dir, "imagesTr")
    labels_dir = os.path.join(raw_dir, "labelsTr")
    for d in (images_dir, labels_dir):
        os.makedirs(d, exist_ok=True)

    datalist = []
    for case_id, ct, pet, label in records:
        _symlink(ct, os.path.join(images_dir, f"{case_id}_0000.nii.gz"))   # canal 0 = CT
        _symlink(pet, os.path.join(images_dir, f"{case_id}_0001.nii.gz"))  # canal 1 = PET (SUV)
        _symlink(label, os.path.join(labels_dir, f"{case_id}.nii.gz"))
        datalist.append({"image": [ct, pet], "label": label, "case_id": case_id})

    # background=0, GTVp=1 (tumeur primaire), GTVn=2 (ganglions) — labels HECKTOR.
    generate_dataset_json(
        output_folder=raw_dir,
        channel_names={0: "CT", 1: "PET"},
        labels={"background": 0, "GTVp": 1, "GTVn": 2},
        num_training_cases=len(records),
        file_ending=".nii.gz",
        dataset_name=config.nnunet_dataset_name,
    )

    datalist_path = os.path.join(config.nnunet_work_dir, "datalist.json")
    with open(datalist_path, "w") as f:
        json.dump({"training": datalist, "testing": []}, f, indent=2)
    print(f"[nnunet] {len(records)} cas → {raw_dir}")
    return datalist_path


def write_project_split(train_ids: list, val_ids: list, valid_case_ids: list):
    """Fige le split déterministe du projet (`split_case_ids`) en fold 0 du
    `splits_final.json` nnU-Net : la validation porte alors exactement sur les mêmes
    patients que le reste de la pipeline. nnU-Net régénère une CV 5-fold s'il est absent."""
    valid = set(valid_case_ids)
    split = [{
        "train": [c for c in train_ids if c in valid],
        "val": [c for c in val_ids if c in valid],
    }]
    preprocessed_dir = os.path.join(config.nnunet_preprocessed_dir, config.nnunet_dataset_dir)
    os.makedirs(preprocessed_dir, exist_ok=True)
    with open(os.path.join(preprocessed_dir, "splits_final.json"), "w") as f:
        json.dump(split, f, indent=2)
    print(f"[nnunet] fold 0 figé : {len(split[0]['train'])} train / {len(split[0]['val'])} val")


def report_validation():
    """Affiche le Dice de validation (moyenne foreground + par classe) écrit par nnU-Net
    à la fin de l'entraînement."""
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


def main():
    for d in (config.experiment_dir, config.nnunet_work_dir, config.nnunet_raw_dir,
              config.nnunet_preprocessed_dir, config.nnunet_results_dir):
        os.makedirs(d, exist_ok=True)

    known_patients = set(pd.read_csv(config.csv_path)["PatientID"])
    train_ids, val_ids = split_case_ids(config)
    records = build_records(train_ids + val_ids, known_patients)
    valid_case_ids = [r[0] for r in records]

    datalist_path = prepare_raw_dataset(records)

    runner = nnUNetV2Runner(
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
        work_dir=config.nnunet_work_dir,
    )

    # Empreinte + planification + prétraitement de la seule configuration entraînée.
    runner.plan_and_process(
        verify_dataset_integrity=True,
        gpu_memory_target=config.nnunet_gpu_memory_gb,
        c=(config.nnunet_configuration,),
        n_proc=(config.num_workers,),
        npfp=config.num_workers,
    )

    if config.nnunet_use_project_split:
        if config.nnunet_fold != 0:
            raise ValueError(
                "nnunet_use_project_split=True impose nnunet_fold=0 (le projet n'a qu'un split)")
        write_project_split(train_ids, val_ids, valid_case_ids)

    runner.train_single_model(
        config=config.nnunet_configuration,
        fold=config.nnunet_fold,
        gpu_id=0,
    )

    report_validation()
    print(f"[nnunet] modèles sauvegardés dans {config.nnunet_results_dir}")


if __name__ == "__main__":
    main()
