"""Construction du jeu de données brut nnU-Net à partir des volumes CT/PET/masque déjà
prétraités, et figement du split du projet.

On contourne `runner.convert_dataset()` (qui exige un unique volume multi-canaux par cas et
renomme les patients en `case_N`) : on crée directement l'arborescence `imagesTr`/`labelsTr`
par liens symboliques (CT → canal 0000, PET → canal 0001), plus `dataset.json`.
"""
import os
import json
import config
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


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
