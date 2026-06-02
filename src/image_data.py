import os
import torch
import pandas as pd
from typing import List
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


# Construit le pipeline de transforms d'entraînement multitâche avec augmentations
def get_train_transforms(config):
    # Clés des modalités images et du masque de segmentation
    keys = ["ct", "pet", "label"]
    # Clés à conserver après le SelectItemsd (image fusionnée, masque, identifiant)
    keep = ["image", "label", "case_id"]

    transforms = [
        # Charge les fichiers .nii.gz via SimpleITK (retourne (H, W, D), ajoute meta_dict)
        LoadImaged(keys=keys, image_only=False, ensure_channel_first=False),
        # Ajoute la dimension canal : (H, W, D) → (1, H, W, D)
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        # Crop aléatoire centré sur les classes tumorales
        RandCropByLabelClassesd(
            keys=keys,
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
        transforms.extend([
            # Flip aléatoire selon les 3 axes
            RandFlipd(keys=keys, spatial_axis=[0, 1, 2], prob=config.aug_probability),
            # Mise à l'échelle aléatoire de l'intensité CT
            RandScaleIntensityd(keys=["ct"], factors=0.1, prob=config.aug_probability),
            # Décalage aléatoire de l'intensité CT
            RandShiftIntensityd(keys=["ct"], offsets=0.1, prob=config.aug_probability),
            # Ajout de bruit gaussien au CT
            RandGaussianNoised(keys=["ct"], std=0.01, prob=config.aug_probability),
            # Lissage gaussien aléatoire du CT pour simuler des résolutions variables
            RandGaussianSmoothd(
                keys=["ct"],
                sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15),
                prob=config.aug_probability,
            ),
        ])

    transforms.extend([
        # Concatène CT et PET sur le canal 0 pour former l'image bimodale (2, D, H, W)
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        # Supprime les clés intermédiaires et ne conserve que les sorties utiles
        SelectItemsd(keys=keep),
        # Assure que image et label sont des tenseurs PyTorch
        EnsureTyped(keys=["image", "label"]),
    ])
    return Compose(transforms)


# Construit le pipeline de transforms de validation multitâche sans augmentation
def get_validation_transforms(config):
    # Clés des modalités et du masque
    keys = ["ct", "pet", "label"]
    # Clés conservées après la sélection (identiques à l'entraînement)
    keep = ["image", "label", "case_id"]
    return Compose([
        # Charge les fichiers .nii.gz
        LoadImaged(keys=keys, image_only=False, ensure_channel_first=False),
        # Ajoute la dimension canal : (H, W, D) → (1, H, W, D)
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        # Crop centré pour garantir la taille spatiale attendue par le réseau
        CenterSpatialCropd(keys=keys, roi_size=config.spatial_size),
        # Padding si le volume est plus petit que spatial_size
        SpatialPadd(keys=keys, spatial_size=config.spatial_size),
        # Fusionne CT et PET en une image bimodale (2, D, H, W)
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        # Conserve uniquement les clés nécessaires
        SelectItemsd(keys=keep),
        # Assure le type tensor PyTorch
        EnsureTyped(keys=["image", "label"]),
    ])


# Construit la liste de dicts d'un split : uniquement chemins d'images + identifiant.
# Les cibles cliniques ne dépendent pas du backbone et sont assemblées séparément
# (voir clinical_data.build_clinical_targets), donc elles ne transitent plus ici.
def _build_data_list(case_ids, data_root, df) -> List[dict]:
    # Patients présents dans le CSV (les autres sont ignorés)
    known = set(df["PatientID"])
    items = []
    for cid in case_ids:
        if cid not in known:
            continue
        # Dossier du patient : {data_root}/{cid}/
        patient_dir = os.path.join(data_root, cid)
        items.append({
            # Chemin vers le fichier CT : {cid}/{cid}__CT.nii.gz
            "ct":      os.path.join(patient_dir, f"{cid}__CT.nii.gz"),
            # Chemin vers le fichier PET : {cid}/{cid}__PT.nii.gz
            "pet":     os.path.join(patient_dir, f"{cid}__PT.nii.gz"),
            # Chemin vers le masque de segmentation : {cid}/{cid}.nii.gz
            "label":   os.path.join(patient_dir, f"{cid}.nii.gz"),
            # Identifiant patient (clé de jointure avec les cibles cliniques)
            "case_id": cid,
        })
    return items


# Découpe les patients en splits train/val de façon déterministe (même seed partout)
def split_case_ids(config) -> tuple:
    import random

    # Découverte des patients : chaque sous-dossier de data_root est un patient
    case_ids = sorted(
        d for d in os.listdir(config.data_root)
        if os.path.isdir(os.path.join(config.data_root, d))
    )

    random.seed(config.seed)
    random.shuffle(case_ids)
    # Nombre de cas réservés à la validation
    n_val = int(len(case_ids) * config.val_split)
    # (train_ids, val_ids)
    return case_ids[n_val:], case_ids[:n_val]


# Construit les listes d'items (images + case_id) des deux splits, partagé par les
# deux points d'entrée (entraînement segmentation et extraction de features).
def _prepare_items(config, tag: str) -> tuple:
    train_ids, val_ids = split_case_ids(config)
    print(f"[{tag}] {len(train_ids)} train / {len(val_ids)} val")
    df = pd.read_csv(config.csv_path)
    return (_build_data_list(train_ids, config.data_root, df),
            _build_data_list(val_ids, config.data_root, df))


# Enveloppe une liste d'items dans un CacheDataset + DataLoader avec les options voulues
def _make_loader(items, transform, config, *, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = CacheDataset(data=items, transform=transform,
                      cache_rate=config.cache_rate, num_workers=config.num_workers)
    return DataLoader(
        ds, batch_size=config.batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )


# Crée les DataLoaders train et val pour l'entraînement de la segmentation
# (augmentation au train, shuffle + drop_last pour des batchs complets).
def get_seg_dataloaders(config) -> tuple:
    train_items, val_items = _prepare_items(config, "Data")
    train_loader = _make_loader(train_items, get_train_transforms(config), config,
                                shuffle=True, drop_last=True)
    val_loader = _make_loader(val_items, get_validation_transforms(config), config,
                              shuffle=False, drop_last=False)
    return train_loader, val_loader


# Crée des loaders déterministes (transforms de validation, sans shuffle) pour
# l'extraction de features : le backbone est figé, on veut un bottleneck reproductible.
# Ne porte que les images + case_id ; les cibles cliniques sont assemblées en phase 2.
def get_feature_extraction_loaders(config) -> tuple:
    train_items, val_items = _prepare_items(config, "Extract")
    tf = get_validation_transforms(config)
    # Sans shuffle ni drop_last : on veut tous les patients, dans l'ordre
    train_loader = _make_loader(train_items, tf, config, shuffle=False, drop_last=False)
    val_loader = _make_loader(val_items, tf, config, shuffle=False, drop_last=False)
    return train_loader, val_loader


# Passe tout un split dans le backbone figé et collecte uniquement les bottlenecks.
# Les case_id permettent de joindre ensuite les cibles cliniques (depuis le CSV).
@torch.no_grad()
def _extract_split(model, loader, device):
    feats, ids = [], []
    for batch in tqdm(loader, desc="Extract", leave=False):
        ct_pet = batch["image"].to(device, non_blocking=True)
        _, bottleneck = model(ct_pet)
        feats.append(bottleneck.float().cpu())
        ids.extend(batch["case_id"])
    return {
        "bottleneck": torch.cat(feats),
        "case_id":    ids,
    }


# Lance le backbone de segmentation figé pour produire les .pt de bottlenecks des
# deux splits (train/val). C'est la phase 1 consommée par l'entraînement clinique.
@torch.no_grad()
def extract_bottlenecks(config, device):
    model = SwinUNETRBackbone(
        input_channels=config.input_channels,
        num_classes=config.num_seg_classes,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        pretrained_path=config.pretrained_path,
    ).to(device)
    # Charge les poids du meilleur modèle de segmentation, puis fige
    model.load_state_dict(torch.load(config.best_seg_path, map_location=device))
    model.eval()

    # Loaders déterministes (transforms de validation, sans shuffle)
    train_loader, val_loader = get_feature_extraction_loaders(config)
    os.makedirs(config.features_dir, exist_ok=True)
    print("extracting train split")
    torch.save(_extract_split(model, train_loader, device), config.train_features_path)
    print("extracting val split")
    torch.save(_extract_split(model, val_loader, device), config.val_features_path)
    print(f"features saved to {config.features_dir}")
