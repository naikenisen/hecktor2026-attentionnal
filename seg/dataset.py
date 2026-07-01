"""Construction du jeu de données brut nnU-Net à partir des volumes CT/PET/masque déjà
prétraités, et figement du split du projet.

On contourne `runner.convert_dataset()` (qui exige un unique volume multi-canaux par cas et
renomme les patients en `case_N`) : on crée directement l'arborescence `imagesTr`/`labelsTr`
par liens symboliques (CT → canal 0000, PET → canal 0001), plus `dataset.json`.
"""
import os
import json
from . import config
from .split import patient_dir
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def build_records(ids_by_split: dict, known_patients: set) -> list:
    """(case_id, ct, pet, label) pour les patients du CSV dont les 3 volumes existent.
    `ids_by_split` associe chaque split ("train"/"test") à ses case_ids : les chemins sont
    résolus dans le sous-dossier du split correspondant."""
    records = []
    skipped = []
    total = 0
    for split, case_ids in ids_by_split.items():
        for case_id in case_ids:
            total += 1
            if case_id not in known_patients:
                skipped.append((case_id, "absent du CSV"))
                continue
            pdir = patient_dir(config, split, case_id)
            ct = os.path.join(pdir, f"{case_id}__CT.nii.gz")
            pet = os.path.join(pdir, f"{case_id}__PT.nii.gz")
            label = os.path.join(pdir, f"{case_id}.nii.gz")
            missing = [m for m, p in [("CT", ct), ("PT", pet), ("label", label)] if not os.path.exists(p)]
            if missing:
                skipped.append((case_id, f"fichier(s) manquant(s): {', '.join(missing)}"))
                continue
            records.append((case_id, ct, pet, label))
    if skipped:
        print(f"[dataset] {len(skipped)} patient(s) ignoré(s) :")
        for pid, reason in skipped:
            print(f"  skip {pid} — {reason}")
    print(f"[dataset] {len(records)} patient(s) retenus sur {total}")
    return records


def _symlink(src: str, dst: str):
    if os.path.lexists(dst):
        os.unlink(dst)
    os.symlink(os.path.abspath(src), dst)


def _clean_symlinks(directory: str):
    """Supprime tous les symlinks existants dans un répertoire."""
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        fp = os.path.join(directory, name)
        if os.path.islink(fp):
            os.unlink(fp)


def prepare_raw_dataset(records: list) -> str:
    """Crée `Dataset<ID>_<name>/{imagesTr,labelsTr}` (liens symboliques) + `dataset.json`,
    et un datalist MONAI (trace + clé requise par l'`input_config` du runner). Retourne le
    chemin du datalist."""
    raw_dir = os.path.join(config.nnunet_raw_dir, config.nnunet_dataset_dir)
    images_dir = os.path.join(raw_dir, "imagesTr")
    labels_dir = os.path.join(raw_dir, "labelsTr")
    for d in (images_dir, labels_dir):
        os.makedirs(d, exist_ok=True)
    _clean_symlinks(images_dir)
    _clean_symlinks(labels_dir)

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

    datalist_path = config.datalist_path
    with open(datalist_path, "w") as f:
        json.dump({"training": datalist, "testing": []}, f, indent=2)
    print(f"[nnunet] {len(records)} cas → {raw_dir}")
    return datalist_path


def write_project_split(train_ids: list, test_ids: list, valid_case_ids: list):
    """Fige le split train/test du disque en fold 0 du `splits_final.json` nnU-Net : la
    validation nnU-Net porte alors exactement sur les mêmes patients (le split test) que le
    reste de la pipeline. nnU-Net régénère une CV 5-fold s'il est absent. La clé `"val"` est
    imposée par le schéma nnU-Net (c'est notre split test)."""
    valid = set(valid_case_ids)
    split = [{
        "train": [c for c in train_ids if c in valid],
        "val": [c for c in test_ids if c in valid],
    }]
    preprocessed_dir = os.path.join(config.nnunet_preprocessed_dir, config.nnunet_dataset_dir)
    os.makedirs(preprocessed_dir, exist_ok=True)
    with open(os.path.join(preprocessed_dir, "splits_final.json"), "w") as f:
        json.dump(split, f, indent=2)
    print(f"[nnunet] fold 0 figé : {len(split[0]['train'])} train / {len(split[0]['val'])} test")
