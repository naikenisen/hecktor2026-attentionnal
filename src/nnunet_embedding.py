"""Extraction et mise en commun de l'embedding du bottleneck de l'encodeur nnU-Net.

Recharge le modèle nnU-Net entraîné (phase 1), GÈLE son encodeur, et sauvegarde le
bottleneck spatial profond de chaque patient (carte du dernier étage d'encodage) vers
`features/{train,val}.pt`. Code **partagé** par les têtes qui exploitent cet embedding :
`tn` (stades T/N) et `nnunet_survival` (survie). `pool_embedding` réduit le bottleneck
spatial en un vecteur par patient.
"""
import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast
from monai.transforms import ResizeWithPadOrCrop
from src.split import split_case_ids


def pool_embedding(bottleneck: torch.Tensor):
    """Réduit le bottleneck spatial (N, C, D, H, W) en un vecteur par patient.
    On concatène moyenne globale (contexte) et maximum global (pic d'activation,
    proche d'un SUVmax) : l'embedding TEP/CT figé servi tel quel aux forêts."""
    mean = bottleneck.mean(dim=(2, 3, 4))
    peak = bottleneck.amax(dim=(2, 3, 4))
    return torch.cat([mean, peak], dim=1).numpy()


class PetCtDataModule:
    """Partitionne les patients en train/val (split déterministe partagé avec le reste de
    la pipeline) et expose les chemins CT/PET/masque de chaque cas."""

    def __init__(self, config):
        self.config = config
        self.known_patients = set(pd.read_csv(config.csv_path)["PatientID"])
        self.train_ids, self.val_ids = self._split_case_ids()

    def _split_case_ids(self) -> tuple:
        train_ids, val_ids = split_case_ids(self.config)
        print(f"{len(train_ids)} train / {len(val_ids)} val")
        return train_ids, val_ids

    def _build_records(self, case_ids: list) -> list:
        records = []
        for case_id in case_ids:
            if case_id not in self.known_patients:
                continue
            patient_dir = os.path.join(self.config.data_root, case_id)
            records.append({
                "ct":      os.path.join(patient_dir, f"{case_id}__CT.nii.gz"),
                "pet":     os.path.join(patient_dir, f"{case_id}__PT.nii.gz"),
                "label":   os.path.join(patient_dir, f"{case_id}.nii.gz"),
                "case_id": case_id,
            })
        return records


class NNUNetBottleneckExtractor:
    """Recharge le modèle nnU-Net entraîné (phase 1), GÈLE son encodeur, et sauvegarde le
    bottleneck spatial profond de chaque patient (carte du dernier étage d'encodage),
    indexé par case_id, pour les deux splits.

    Chaque cas est prétraité par nnU-Net lui-même (resampling à l'espacement cible +
    normalisation par canal, identiques à l'entraînement) puis recadré au patch fixe du
    plan : le bottleneck a donc une taille spatiale fixe, empilable. Le format de sortie
    (`{"bottleneck": (N, C, D, H, W), "case_id": [...]}`) est consommé via `pool_embedding`
    par les forêts T/N et de survie."""

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
        os.makedirs(self.config.features_dir, exist_ok=True)
        for split_ids, out_path, name in (
            (self.data.train_ids, self.config.train_features_path, "train"),
            (self.data.val_ids, self.config.val_features_path, "val"),
        ):
            records = self.data._build_records(split_ids)
            print(f"extracting {name} split ({len(records)} cases)")
            torch.save(self._bottlenecks_of(predictor, encoder, preprocessor, crop, records), out_path)
        print(f"features saved to {self.config.features_dir}")


def ensure_bottlenecks(config):
    """Extrait les embeddings du bottleneck de l'encodeur nnU-Net si les fichiers de
    features sont absents. Idempotent (utilisé par `tn.train` et `nnunet_survival.train`) :
    n'utilise le GPU que lors de la première extraction."""
    if os.path.exists(config.train_features_path) and os.path.exists(config.val_features_path):
        print("bottleneck features already extracted, skipping")
        return
    device = torch.device("cuda")
    NNUNetBottleneckExtractor(config, device).run()
