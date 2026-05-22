import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple
from monai.networks.nets import SwinUNETR

# Configuration du backbone SwinUNETR utilisée par SwinUNETRMultitask
@dataclass
class SwinUNETRConfig:

    # Chemin racine vers les données d'entraînement
    data_root: str = "/path/to/hecktor2026_training"
    # Sous-dossier des images CT/PET preprocessées
    train_images_dir: str = "imagesTr_resampled_cropped_npy"
    # Sous-dossier des labels de segmentation preprocessés
    train_labels_dir: str = "labelsTr_resampled_cropped_npy"

    # Nombre de canaux d'entrée (CT + PET)
    input_channels: int = 2
    # Nombre de classes de segmentation (bg, GTVp, GTVn)
    num_classes: int = 3
    # Taille spatiale du volume 3D en voxels
    spatial_size: Tuple[int, int, int] = (128, 128, 128)

    # Taille de batch
    batch_size: int = 2
    # Learning rate initial
    learning_rate: float = 1e-4
    # Coefficient de régularisation L2
    weight_decay: float = 1e-5
    # Nombre total d'epochs
    num_epochs: int = 350
    # Puissance du scheduler PolynomialLR
    poly_lr_power: float = 0.9
    # Learning rate minimum pour le scheduler
    poly_lr_min_lr: float = 1e-6

    # Active les augmentations d'entraînement
    use_augmentation: bool = True
    # Probabilité d'application de chaque augmentation
    aug_probability: float = 0.5

    # Périphérique cible
    device: str = "cuda"
    # Nombre de workers du DataLoader
    num_workers: int = 4
    # Fraction du dataset mise en cache RAM
    cache_rate: float = 0.25
    # Fréquence de sauvegarde en epochs
    save_checkpoint_every: int = 1
    # Active les logs TensorBoard
    use_tensorboard: bool = True

    # Nom de l'expérience pour l'arborescence de sortie
    experiment_name: str = "swinunetr"
    # Dossier racine de sortie
    output_dir: str = "experiments"

    # Taille des feature maps (doit correspondre aux poids SSL MONAI)
    feature_size: int = 48
    # Active le gradient checkpointing pour économiser la VRAM
    use_checkpoint: bool = True

    # Chemin vers les poids SSL pré-entraînés du SwinViT MONAI
    pretrained_path: str = "/beegfs/data/work/imvia/in156281/hecktor2026-attentionnal/utils/model_swinvit.pt"

    # Crée les dossiers de sortie après initialisation du dataclass
    def __post_init__(self):
        # Dossier principal de l'expérience
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        # Dossier des checkpoints sauvegardés
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        # Dossier des logs TensorBoard
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        for d in [self.experiment_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)


# Sous-classe MONAI SwinUNETR qui expose le bottleneck en plus du masque de segmentation
class SwinUNETRMultitask(nn.Module):

    # Instancie le SwinUNETR MONAI et charge les poids SSL si disponibles
    def __init__(self, config: SwinUNETRConfig):
        super().__init__()
        # Référence à la configuration du backbone
        self.config = config

        # Instance SwinUNETR MONAI standard (sera appelé manuellement dans forward)
        self.swinunetr = SwinUNETR(
            in_channels=config.input_channels,
            out_channels=config.num_classes,
            feature_size=config.feature_size,
            use_checkpoint=config.use_checkpoint,
        )

        if config.pretrained_path and os.path.exists(config.pretrained_path):
            weights = torch.load(config.pretrained_path, map_location="cpu", weights_only=False)
            self.swinunetr.load_from(weights=weights)
            print(f"[SwinUNETRMultitask] SSL weights loaded from '{config.pretrained_path}'.")
        else:
            print(f"[SwinUNETRMultitask] WARNING: SSL weights not found ({config.pretrained_path}). Random init.")

    # Retourne (seg_logits, bottleneck) en reproduisant manuellement le forward MONAI
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Raccourci vers l'instance SwinUNETR interne
        net = self.swinunetr
        # Liste des feature maps hiérarchiques [stage0..stage3, bottleneck]
        hidden_states_out = net.swinViT(x, net.normalize)
        # Feature map la plus profonde (downscale ×32, C=feature_size×16)
        bottleneck = hidden_states_out[4]

        # Encodeur skip-connection niveau 0 (résolution originale)
        enc0 = net.encoder1(x)
        # Encodeur skip-connection niveau 1
        enc1 = net.encoder2(hidden_states_out[0])
        # Encodeur skip-connection niveau 2
        enc2 = net.encoder3(hidden_states_out[1])
        # Encodeur skip-connection niveau 3
        enc3 = net.encoder4(hidden_states_out[2])
        # Encodeur bottleneck (niveau 4)
        dec4 = net.encoder10(hidden_states_out[4])
        # Décodeur niveau 3
        dec3 = net.decoder5(dec4, hidden_states_out[3])
        # Décodeur niveau 2
        dec2 = net.decoder4(dec3, enc3)
        # Décodeur niveau 1
        dec1 = net.decoder3(dec2, enc2)
        # Décodeur niveau 0
        dec0 = net.decoder2(dec1, enc1)
        # Décodeur final avant la couche de sortie
        out = net.decoder1(dec0, enc0)
        # Logits de segmentation (B, num_classes, D, H, W)
        seg_logits = net.out(out)

        return seg_logits, bottleneck
