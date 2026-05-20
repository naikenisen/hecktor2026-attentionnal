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


# Construit le pipeline de transforms d'entraînement multitâche avec augmentations
def get_multitask_train_transforms(config):
    # Clés des modalités images et du masque de segmentation
    keys = ["ct", "pet", "label"]
    # Clés à conserver après le SelectItemsd (inclut les cibles tabulaires)
    keep = ["image", "label", "clinical", "t_label", "n_label", "time", "event", "case_id"]

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
def get_multitask_validation_transforms(config):
    # Clés des modalités et du masque
    keys = ["ct", "pet", "label"]
    # Clés conservées après la sélection (identiques à l'entraînement)
    keep = ["image", "label", "clinical", "t_label", "n_label", "time", "event", "case_id"]
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
