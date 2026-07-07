"""Phase 1 — point d'entrée de la segmentation nnU-Net.

Lancement (depuis la racine du dépôt) :

    python -m seg.train

nnU-Net est auto-configurant : il dérive patch size, batch size, normalisation et data
augmentation de l'empreinte du jeu de données. Il n'y a donc ni recherche d'hyperparamètres
ni retrain séparé — cette phase est complète à elle seule.

Orchestration : dossier train du disque → arborescence brute nnU-Net → plan_and_process →
entraîne 3d_fullres (nnU-Net tire lui-même son split train/validation par CV interne, sans le
test) → reporte le Dice de validation. Le test set n'intervient pas dans cette phase — il reste
réservé à l'évaluation finale.

L'extraction du bottleneck de l'encodeur fait l'objet d'un script SÉPARÉ, `seg.extract`, à
lancer ensuite (`python -m seg.extract`).
"""
import os
import pandas as pd
from . import config
from .split import case_ids
from seg.dataset import build_records, prepare_raw_dataset
from seg.runner import build_runner, plan_and_process, train_fold, report_validation


def main():
    for d in (config.results_dir, config.nnunet_raw_dir,
              config.nnunet_preprocessed_dir, config.nnunet_results_dir):
        os.makedirs(d, exist_ok=True)

    known_patients = set(pd.read_csv(config.csv_path)["PatientID"])
    train_ids = case_ids(config, "train")
    records = build_records({"train": train_ids}, known_patients)

    datalist_path = prepare_raw_dataset(records)
    runner = build_runner(datalist_path)
    plan_and_process(runner)

    train_fold(runner)
    report_validation()
    print(f"[nnunet] meilleur poids sauvegardé : {config.nnunet_model_dir}/fold_{config.nnunet_fold}/checkpoint_best.pth")


if __name__ == "__main__":
    main()
