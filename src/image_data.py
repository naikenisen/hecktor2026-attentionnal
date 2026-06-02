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

def get_train_transforms(config):
    keys = ["ct", "pet", "label"]
    keep = ["image", "label", "case_id"]
    transforms = [
        LoadImaged(keys=keys, image_only=False, ensure_channel_first=False),
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
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
            RandFlipd(keys=keys, spatial_axis=[0, 1, 2], prob=config.aug_probability),
            RandScaleIntensityd(keys=["ct"], factors=0.1, prob=config.aug_probability),
            RandShiftIntensityd(keys=["ct"], offsets=0.1, prob=config.aug_probability),
            RandGaussianNoised(keys=["ct"], std=0.01, prob=config.aug_probability),
            RandGaussianSmoothd(
                keys=["ct"],
                sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15),
                prob=config.aug_probability,
            ),
        ])
    transforms.extend([
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        SelectItemsd(keys=keep),
        EnsureTyped(keys=["image", "label"]),
    ])
    return Compose(transforms)

def get_validation_transforms(config):
    keys = ["ct", "pet", "label"]
    keep = ["image", "label", "case_id"]
    return Compose([
        LoadImaged(keys=keys, image_only=False, ensure_channel_first=False),
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        CenterSpatialCropd(keys=keys, roi_size=config.spatial_size),
        SpatialPadd(keys=keys, spatial_size=config.spatial_size),
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        SelectItemsd(keys=keep),
        EnsureTyped(keys=["image", "label"]),
    ])

def _build_data_list(case_ids, data_root, df) -> List[dict]:
    known = set(df["PatientID"])
    items = []
    for cid in case_ids:
        if cid not in known:
            continue
        patient_dir = os.path.join(data_root, cid)
        items.append({
            "ct":      os.path.join(patient_dir, f"{cid}__CT.nii.gz"),
            "pet":     os.path.join(patient_dir, f"{cid}__PT.nii.gz"),
            "label":   os.path.join(patient_dir, f"{cid}.nii.gz"),
            "case_id": cid,
        })
    return items

def split_case_ids(config) -> tuple:
    import random

    case_ids = sorted(
        d for d in os.listdir(config.data_root)
        if os.path.isdir(os.path.join(config.data_root, d))
    )
    random.seed(config.seed)
    random.shuffle(case_ids)
    n_val = int(len(case_ids) * config.val_split)
    return case_ids[n_val:], case_ids[:n_val]

def _prepare_items(config, tag: str) -> tuple:
    train_ids, val_ids = split_case_ids(config)
    print(f"[{tag}] {len(train_ids)} train / {len(val_ids)} val")
    df = pd.read_csv(config.csv_path)
    return (_build_data_list(train_ids, config.data_root, df),
            _build_data_list(val_ids, config.data_root, df))

def _make_loader(items, transform, config, *, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = CacheDataset(data=items, transform=transform,
                      cache_rate=config.cache_rate, num_workers=config.num_workers)
    return DataLoader(
        ds, batch_size=config.batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

def get_seg_dataloaders(config) -> tuple:
    train_items, val_items = _prepare_items(config, "Data")
    train_loader = _make_loader(train_items, get_train_transforms(config), config,
                                shuffle=True, drop_last=True)
    val_loader = _make_loader(val_items, get_validation_transforms(config), config,
                              shuffle=False, drop_last=False)
    return train_loader, val_loader

def get_feature_extraction_loaders(config) -> tuple:
    train_items, val_items = _prepare_items(config, "Extract")
    tf = get_validation_transforms(config)
    train_loader = _make_loader(train_items, tf, config, shuffle=False, drop_last=False)
    val_loader = _make_loader(val_items, tf, config, shuffle=False, drop_last=False)
    return train_loader, val_loader

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

@torch.no_grad()
def extract_bottlenecks(config, device):
    model = SwinUNETRBackbone(
        input_channels=config.input_channels,
        num_classes=config.num_seg_classes,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        pretrained_path=config.pretrained_path,
    ).to(device)
    model.load_state_dict(torch.load(config.best_seg_path, map_location=device))
    model.eval()
    train_loader, val_loader = get_feature_extraction_loaders(config)
    os.makedirs(config.features_dir, exist_ok=True)
    print("extracting train split")
    torch.save(_extract_split(model, train_loader, device), config.train_features_path)
    print("extracting val split")
    torch.save(_extract_split(model, val_loader, device), config.val_features_path)
    print(f"features saved to {config.features_dir}")
