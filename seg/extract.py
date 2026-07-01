"""Phase 1bis — extraction du bottleneck de l'encodeur nnU-Net (script SÉPARÉ).

À lancer sur le cluster APRÈS la segmentation (`python -m seg.train`) :

    python -m seg.extract

Recharge le modèle nnU-Net entraîné, **GÈLE son encodeur**, prétraite chaque patient *via
nnU-Net lui-même* (resampling + normalisation par canal), recadre au patch du plan, réduit le
bottleneck du dernier étage d'encodage en un vecteur par patient (moyenne ⊕ max) et écrit un
CSV **unique** `tables/bottleneck.csv` (colonne `PatientID` + une colonne `feat_i` par
dimension), **tous splits confondus**.

C'est le SEUL endroit du dépôt qui extrait le bottleneck : `tn` et `nnunet_survival` ne font
plus que **lire** ce CSV (croisé avec la colonne `split` du CSV clinique). Le CSV produit sur
le cluster est ensuite rapatrié dans `tables/`.
"""
import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast
from monai.transforms import ResizeWithPadOrCrop

from . import config
from .split import case_ids, patient_dir

FEATURE_PREFIX = "feat_"


def pool_embedding(bottleneck: torch.Tensor):
    """Réduit le bottleneck spatial (N, C, D, H, W) en un vecteur par patient.
    On concatène moyenne globale (contexte) et maximum global (pic d'activation,
    proche d'un SUVmax) : l'embedding TEP/CT figé servi tel quel aux forêts."""
    mean = bottleneck.mean(dim=(2, 3, 4))
    peak = bottleneck.amax(dim=(2, 3, 4))
    return torch.cat([mean, peak], dim=1).numpy()


def write_features(path: str, case_ids: list, embeddings):
    """Écrit les features poolées au format CSV : colonne `PatientID` + une colonne
    `feat_i` par dimension de l'embedding."""
    columns = [f"{FEATURE_PREFIX}{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=columns)
    df.insert(0, "PatientID", case_ids)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


class PetCtDataModule:
    """Rassemble TOUS les patients (train + test, arborescence du disque) et expose les
    chemins CT/PET/masque de chaque cas — un seul CSV de features en sortie."""

    def __init__(self, config):
        self.config = config
        self.known_patients = set(pd.read_csv(config.csv_path)["PatientID"])
        self.records = (self._build_records("train", case_ids(config, "train"))
                        + self._build_records("test", case_ids(config, "test")))

    def _build_records(self, split: str, ids: list) -> list:
        records = []
        for case_id in ids:
            if case_id not in self.known_patients:
                continue
            pdir = patient_dir(self.config, split, case_id)
            records.append({
                "ct":      os.path.join(pdir, f"{case_id}__CT.nii.gz"),
                "pet":     os.path.join(pdir, f"{case_id}__PT.nii.gz"),
                "label":   os.path.join(pdir, f"{case_id}.nii.gz"),
                "case_id": case_id,
            })
        return records


class NNUNetBottleneckExtractor:
    """Recharge le modèle nnU-Net entraîné (phase 1), GÈLE son encodeur, et sauvegarde le
    bottleneck spatial profond de chaque patient (carte du dernier étage d'encodage), réduit
    en un vecteur (`pool_embedding`) et écrit en CSV.

    Chaque cas est prétraité par nnU-Net lui-même (resampling à l'espacement cible +
    normalisation par canal, identiques à l'entraînement) puis recadré au patch fixe du
    plan : le bottleneck a donc une taille spatiale fixe, empilable."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.data = PetCtDataModule(config)

    def _load_predictor(self):
        """nnUNetPredictor : reconstruit l'architecture + charge plans/dataset.json du
        modèle entraîné. Les poids du fold sont dans `list_of_parameters`."""
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(
            perform_everything_on_device=True, device=self.device,
            verbose=False, verbose_preprocessing=False, allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            self.config.nnunet_model_dir,
            use_folds=(self.config.nnunet_fold,),
            checkpoint_name=self.config.nnunet_checkpoint,
        )
        return predictor

    def _build_encoder(self, predictor):
        network = getattr(predictor.network, "_orig_mod", predictor.network)  # déballe torch.compile
        network.load_state_dict(predictor.list_of_parameters[0])
        network.to(self.device).eval()
        return network.encoder  # forward → liste des skips ; le dernier est le bottleneck

    @torch.no_grad()
    def _bottlenecks_of(self, predictor, encoder, preprocessor, crop, records: list) -> dict:
        bottlenecks, case_ids = [], []
        for rec in tqdm(records, desc="Extract", leave=False):
            # Prétraitement nnU-Net (resampling + normalisation par canal) — comme à l'entraînement.
            data, _, _ = preprocessor.run_case(
                [rec["ct"], rec["pet"]], None,
                predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json)
            x = crop(torch.as_tensor(data)).unsqueeze(0).to(self.device, non_blocking=True)
            with autocast():
                bottleneck = encoder(x)[-1]                # (1, C, d, d, d)
            b = bottleneck.detach().float().cpu()
            bottlenecks.append(b.as_tensor() if hasattr(b, "as_tensor") else b)  # strip MetaTensor
            case_ids.append(rec["case_id"])
        return {"bottleneck": torch.cat(bottlenecks), "case_id": case_ids}

    @torch.no_grad()
    def run(self):
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
        predictor = self._load_predictor()
        encoder = self._build_encoder(predictor)
        preprocessor = DefaultPreprocessor(verbose=False)
        crop = ResizeWithPadOrCrop(spatial_size=tuple(predictor.configuration_manager.patch_size))
        records = self.data.records
        print(f"extracting bottleneck for {len(records)} patients (train + test)")
        out = self._bottlenecks_of(predictor, encoder, preprocessor, crop, records)
        write_features(self.config.bottleneck_csv_path, out["case_id"], pool_embedding(out["bottleneck"]))
        print(f"bottleneck features saved to {self.config.bottleneck_csv_path}")


def main():
    NNUNetBottleneckExtractor(config, torch.device("cuda")).run()


if __name__ == "__main__":
    main()
