import numpy as np
import torch
from monai.transforms import (
    Compose,
    MapTransform,
    EnsureChannelFirstd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    EnsureTyped,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    ConcatItemsd,
    SelectItemsd,
)

# Transform MONAI qui charge un fichier .npz et en extrait l'image et les métadonnées
class LoadNpzDictd(MapTransform):

    # Charge chaque clé depuis son fichier .npz et injecte tensor + meta_dict dans le sample
    def __call__(self, data):
        # Copie du dictionnaire pour éviter de modifier l'original
        d = dict(data)
        for key in self.keys:
            # Chemin du fichier .npz pour cette clé
            filepath = d[key]
            # Chargement avec allow_pickle=True requis pour les meta dicts NumPy
            loaded = np.load(filepath, allow_pickle=True)
            # Tableau numpy de l'image ou du masque
            image_array = loaded['image']
            # Dictionnaire de métadonnées spatiales (spacing, origin, affine...)
            meta_dict = loaded['meta'].item()
            # Conversion en tensor PyTorch
            d[key] = torch.from_numpy(image_array)
            # Métadonnées au format attendu par MONAI ({key}_meta_dict)
            d[f"{key}_meta_dict"] = meta_dict
        return d


# Construit le pipeline de transforms d'entraînement multitâche avec augmentations
def get_multitask_train_transforms(config):
    # Clés des modalités images et du masque de segmentation
    keys = ["ct", "pet", "label"]
    # Clés à conserver après le SelectItemsd (inclut les cibles tabulaires)
    keep = ["image", "label", "clinical", "t_label", "n_label", "time", "event", "case_id"]

    # Liste de transforms initialisée avec le chargeur .npz
    transforms = [LoadNpzDictd(keys=keys)]

    # Crop aléatoire centré sur les classes tumorales pour augmenter l'exposition aux lésions
    transforms.append(
        RandCropByLabelClassesd(
            keys=keys,
            label_key="label",
            spatial_size=config.spatial_size,
            ratios=[0.1, 0.45, 0.45],
            num_classes=3,
            num_samples=1,
            allow_missing_keys=True,
            warn=False,
        )
    )

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


# Construit le pipeline de transforms de validation multitâche sans augmentation ni crop aléatoire
def get_multitask_validation_transforms():
    # Clés des modalités et du masque
    keys = ["ct", "pet", "label"]
    # Clés conservées après la sélection (identiques à l'entraînement)
    keep = ["image", "label", "clinical", "t_label", "n_label", "time", "event", "case_id"]
    return Compose([
        # Charge les fichiers .npz
        LoadNpzDictd(keys=keys),
        # Fusionne CT et PET en une image bimodale (2, D, H, W)
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        # Conserve uniquement les clés nécessaires
        SelectItemsd(keys=keep),
        # Assure le type tensor PyTorch
        EnsureTyped(keys=["image", "label"]),
    ])
