"""Phase 1 — point d'entrée de la segmentation nnU-Net.

Lancement (depuis la racine du dépôt) :

    python -m seg.train

nnU-Net est auto-configurant : il dérive patch size, batch size, normalisation et data
augmentation de l'empreinte du jeu de données. Il n'y a donc ni recherche d'hyperparamètres
ni retrain séparé — cette phase est complète à elle seule.

Orchestration : split train/test du disque → arborescence brute nnU-Net → plan_and_process →
fige le split du projet en fold 0 → entraîne 3d_fullres → reporte le Dice de validation.

L'extraction du bottleneck de l'encodeur fait l'objet d'un script SÉPARÉ, `seg.extract`, à
lancer ensuite (`python -m seg.extract`).
"""
import os
import pandas as pd
from . import config
from .split import case_ids
from seg.dataset import build_records, prepare_raw_dataset, write_project_split
from seg.runner import build_runner, plan_and_process, train_fold, report_validation


def main():
    for d in (config.results_dir, config.nnunet_raw_dir,
              config.nnunet_preprocessed_dir, config.nnunet_results_dir):
        os.makedirs(d, exist_ok=True)

    known_patients = set(pd.read_csv(config.csv_path)["PatientID"])
    train_ids = case_ids(config, "train")
    test_ids = case_ids(config, "test")
    records = build_records({"train": train_ids, "test": test_ids}, known_patients)
    valid_case_ids = [r[0] for r in records]

    datalist_path = prepare_raw_dataset(records)
    runner = build_runner(datalist_path)
    plan_and_process(runner)

    if config.nnunet_use_project_split:
        if config.nnunet_fold != 0:
            raise ValueError(
                "nnunet_use_project_split=True impose nnunet_fold=0 (le projet n'a qu'un split)")
        write_project_split(train_ids, test_ids, valid_case_ids)

    train_fold(runner)
    report_validation()
    print(f"[nnunet] modèles sauvegardés dans {config.nnunet_results_dir}")


if __name__ == "__main__":
    main()
