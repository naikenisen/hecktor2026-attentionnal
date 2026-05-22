import os
from dataclasses import dataclass, field
from typing import Tuple

# Centralise tous les hyperparamètres de la pipeline multitâche 2026
@dataclass
class MultitaskConfig:

    # Racine du dataset HECKTOR sur disque (un sous-dossier par patient)
    data_root: str = "/work/imvia/in156281/datasets/hecktor_dataset_preprocessed"
    # Chemin vers le CSV des données cliniques et cibles
    csv_path: str = "/work/imvia/in156281/datasets/hecktor_dataset/HECKTOR_2026_training_data.csv"

    # Nombre de canaux d'entrée (CT + PET)
    input_channels: int = 2
    # Nombre de classes de segmentation (bg, GTVp, GTVn)
    num_seg_classes: int = 3
    # Nombre de classes pour le staging T (T1..T4)
    num_t_classes: int = 4
    # Nombre de classes pour le staging N (N0..N3)
    num_n_classes: int = 4
    # Taille spatiale du volume 3D en entrée du réseau
    spatial_size: Tuple[int, int, int] = (128, 128, 128)

    # Taille des feature maps (doit correspondre aux poids SSL MONAI)
    feature_size: int = 48
    # Active le gradient checkpointing pour économiser la VRAM
    use_checkpoint: bool = True
    # Chemin vers les poids SSL pré-entraînés du SwinViT
    pretrained_path: str = "/beegfs/data/work/imvia/in156281/hecktor2026-attentionnal/model_swinvit.pt"
    # Nombre de canaux du bottleneck SwinUNETR (feature_size=48 → 768)
    bottleneck_channels: int = 768

    # Dimension du token pour la cross-attention et la tête survie
    d_model: int = 256
    # Nombre de têtes de l'attention multi-têtes
    n_heads: int = 4
    # Dimension du vecteur clinique brut en entrée du MLP
    n_clinical_features: int = 7
    # Dimension cachée des têtes T/N après le GAP
    hidden_tn: int = 256

    # Nombre d'intervalles temporels pour la survie discrète
    n_time_bins: int = 10
    # Dimension cachée de la tête survie
    surv_hidden: int = 256

    # Taille de batch pour l'entraînement
    batch_size: int = 2
    # Learning rate initial pour AdamW
    learning_rate: float = 1e-4
    # Régularisation L2 pour AdamW
    weight_decay: float = 1e-5
    # Nombre total d'epochs d'entraînement
    num_epochs: int = 350
    # Nombre d'epochs de warm-up (seule la segmentation est entraînée)
    n_warmup: int = 20
    # Norme maximale pour le gradient clipping
    grad_clip_norm: float = 1.0
    # Puissance du scheduler PolynomialLR
    poly_lr_power: float = 0.9

    # Active les augmentations aléatoires à l'entraînement
    use_augmentation: bool = True
    # Probabilité d'application de chaque augmentation
    aug_probability: float = 0.5

    # Fraction des données réservée à la validation
    val_split: float = 0.2
    # Graine aléatoire pour la reproductibilité
    seed: int = 42

    # Périphérique cible (cuda ou cpu)
    device: str = "cuda"
    # Nombre de workers pour le DataLoader
    num_workers: int = 4
    # Fraction du dataset mise en cache en RAM par MONAI CacheDataset
    cache_rate: float = 0.25
    # Fréquence de sauvegarde des checkpoints intermédiaires (en epochs)
    save_checkpoint_every: int = 5
    # Active l'écriture des métriques dans TensorBoard
    use_tensorboard: bool = True

    # Nom de l'expérience (crée un sous-dossier dédié)
    experiment_name: str = "multitask_e2e"
    # Dossier racine de sortie pour les expériences
    output_dir: str = "experiments"

    # Crée automatiquement les dossiers de sortie après initialisation
    def __post_init__(self):
        # Dossier principal de l'expérience
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        # Dossier des checkpoints
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        # Dossier des logs TensorBoard
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        for d in [self.experiment_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)
