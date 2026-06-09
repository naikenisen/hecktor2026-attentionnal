import os
import random
import torch
import pandas as pd

# Les workers du DataLoader partagent les tensors via des file descriptors par défaut ;
# sur de gros volumes bimodaux on épuise vite `ulimit -n` (RuntimeError: received 0 items
# of ancdata). On bascule sur le partage par mémoire partagée (/dev/shm).
torch.multiprocessing.set_sharing_strategy("file_system")
from tqdm import tqdm
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    SpatialPadd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    EnsureTyped,
    RandCropByLabelClassesd,
    ConcatItemsd,
    SelectItemsd,
)
from src.networks import SwinUNETRBackbone

VOLUME_KEYS = ["ct", "pet", "label"]
MODALITY_KEYS = ["ct", "pet"]
SAMPLE_KEYS = ["image", "label", "case_id"]

class PetCtTransforms:
    """Fabrique les pipelines MONAI qui chargent CT+PET+masque et les fusionnent en
    volume bimodal : augmenté à l'entraînement, déterministe en validation."""

    @staticmethod
    def train(config) -> Compose:
        steps = [
            LoadImaged(keys=VOLUME_KEYS, image_only=False, ensure_channel_first=False),
            EnsureChannelFirstd(keys=VOLUME_KEYS, channel_dim="no_channel"),
            RandCropByLabelClassesd(
                keys=VOLUME_KEYS,
                label_key="label",
                spatial_size=config.spatial_size,
                ratios=[0.1, 0.45, 0.45],
                num_classes=3,
                num_samples=1,
                allow_missing_keys=True,
                warn=False,
            ),
        ]
        if config.use_augmentation:
            steps.extend([
                RandFlipd(keys=VOLUME_KEYS, spatial_axis=[0, 1, 2], prob=config.aug_probability),
                RandScaleIntensityd(keys=["ct"], factors=0.1, prob=config.aug_probability),
                RandShiftIntensityd(keys=["ct"], offsets=0.1, prob=config.aug_probability),
                RandGaussianNoised(keys=["ct"], std=0.01, prob=config.aug_probability),
                RandGaussianSmoothd(
                    keys=["ct"],
                    sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15),
                    prob=config.aug_probability,
                ),
            ])
        steps.extend([
            ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
            SelectItemsd(keys=SAMPLE_KEYS),
            EnsureTyped(keys=["image", "label"]),
        ])
        return Compose(steps)

    @staticmethod
    def validation(config) -> Compose:
        # Crop central 128³ : conservé pour l'extraction de features (bottleneck figé de taille fixe)
        return Compose([
            LoadImaged(keys=VOLUME_KEYS, image_only=False, ensure_channel_first=False),
            EnsureChannelFirstd(keys=VOLUME_KEYS, channel_dim="no_channel"),
            CenterSpatialCropd(keys=VOLUME_KEYS, roi_size=config.spatial_size),
            SpatialPadd(keys=VOLUME_KEYS, spatial_size=config.spatial_size),
            ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
            SelectItemsd(keys=SAMPLE_KEYS),
            EnsureTyped(keys=["image", "label"]),
        ])

    @staticmethod
    def segmentation_validation(config) -> Compose:
        # Volume entier passé tel quel : le sliding_window_inference se charge du pavage en 128³
        return Compose([
            LoadImaged(keys=VOLUME_KEYS, image_only=False, ensure_channel_first=False),
            EnsureChannelFirstd(keys=VOLUME_KEYS, channel_dim="no_channel"),
            ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
            SelectItemsd(keys=SAMPLE_KEYS),
            EnsureTyped(keys=["image", "label"]),
        ])

class PetCtDataModule:
    """Partitionne les patients en train/val et construit les DataLoaders cachés, aussi
    bien pour l'entraînement de la segmentation que pour l'extraction des features."""

    def __init__(self, config):
        self.config = config
        self.known_patients = set(pd.read_csv(config.csv_path)["PatientID"])
        self.train_ids, self.val_ids = self._split_case_ids()

    def _split_case_ids(self) -> tuple:
        case_ids = sorted(
            d for d in os.listdir(self.config.data_root)
            if os.path.isdir(os.path.join(self.config.data_root, d))
        )
        random.seed(self.config.seed)
        random.shuffle(case_ids)
        n_val = int(len(case_ids) * self.config.val_split)
        train_ids, val_ids = case_ids[n_val:], case_ids[:n_val]
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

    def _build_loader(self, case_ids: list, transforms: Compose, *, shuffle: bool,
                      drop_last: bool, batch_size: int = None) -> DataLoader:
        dataset = CacheDataset(
            data=self._build_records(case_ids), transform=transforms,
            cache_rate=self.config.cache_rate, num_workers=self.config.num_workers,
        )
        return DataLoader(
            dataset, batch_size=batch_size or self.config.batch_size, shuffle=shuffle,
            drop_last=drop_last, num_workers=self.config.num_workers, pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def segmentation_loaders(self) -> tuple:
        train_loader = self._build_loader(self.train_ids, PetCtTransforms.train(self.config),
                                    shuffle=True, drop_last=True)
        # Val sur volume entier : batch_size=1 (tailles variables possibles) + sliding window
        val_loader = self._build_loader(self.val_ids, PetCtTransforms.segmentation_validation(self.config),
                                  shuffle=False, drop_last=False, batch_size=1)
        return train_loader, val_loader

    def extraction_loaders(self) -> tuple:
        transforms = PetCtTransforms.validation(self.config)
        train_loader = self._build_loader(self.train_ids, transforms, shuffle=False, drop_last=False)
        val_loader = self._build_loader(self.val_ids, transforms, shuffle=False, drop_last=False)
        return train_loader, val_loader

class BottleneckExtractor:
    """Recharge le backbone de segmentation figé et sauvegarde sur disque le bottleneck
    de chaque patient, indexé par case_id, pour les deux splits."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.data = PetCtDataModule(config)

    def _load_frozen_backbone(self) -> SwinUNETRBackbone:
        backbone = SwinUNETRBackbone(
            input_channels=self.config.input_channels,
            num_classes=self.config.num_seg_classes,
            feature_size=self.config.feature_size,
            use_checkpoint=self.config.use_checkpoint,
            pretrained_path=self.config.pretrained_path,
        ).to(self.device)
        backbone.load_state_dict(torch.load(self.config.best_seg_path, map_location=self.device))
        backbone.eval()
        return backbone

    @torch.no_grad()
    def _bottlenecks_of(self, backbone: SwinUNETRBackbone, loader: DataLoader) -> dict:
        bottlenecks, case_ids = [], []
        for batch in tqdm(loader, desc="Extract", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            _, bottleneck = backbone(images)
            # .as_tensor() retire les métadonnées MONAI (MetaTensor) : le fichier de
            # features reste un torch.Tensor pur, chargeable avec weights_only=True.
            bottlenecks.append(bottleneck.as_tensor().float().cpu())
            case_ids.extend(batch["case_id"])
        return {"bottleneck": torch.cat(bottlenecks), "case_id": case_ids}

    @torch.no_grad()
    def run(self):
        backbone = self._load_frozen_backbone()
        train_loader, val_loader = self.data.extraction_loaders()
        os.makedirs(self.config.features_dir, exist_ok=True)
        print("extracting train split")
        torch.save(self._bottlenecks_of(backbone, train_loader), self.config.train_features_path)
        print("extracting val split")
        torch.save(self._bottlenecks_of(backbone, val_loader), self.config.val_features_path)
        print(f"features saved to {self.config.features_dir}")


def ensure_bottlenecks(config):
    """Extrait les embeddings TEP/CT figés si les fichiers de features sont absents.
    Idempotent : partagé par train_tn.py et train_survival.py, n'utilise le GPU
    que lors de la première extraction."""
    if os.path.exists(config.train_features_path) and os.path.exists(config.val_features_path):
        print("bottleneck features already extracted, skipping")
        return
    device = torch.device("cuda")
    BottleneckExtractor(config, device).run()
