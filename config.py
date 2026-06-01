# Racine du dataset HECKTOR sur disque (un sous-dossier par patient)
data_root = "/work/imvia/in156281/datasets/hecktor_dataset_preprocessed"
# Chemin vers le CSV des données cliniques et cibles
csv_path = "/work/imvia/in156281/datasets/hecktor_dataset/HECKTOR_2026_training_data.csv"

# Nombre de canaux d'entrée (CT + PET)
input_channels = 2
# Nombre de classes de segmentation (bg, GTVp, GTVn)
num_seg_classes = 3
# Nombre de classes pour le staging T (T1..T4)
num_t_classes = 4
# Nombre de classes pour le staging N (N0..N3)
num_n_classes = 3
# Taille spatiale du volume 3D en entrée du réseau
spatial_size = (128, 128, 128)

# Taille des feature maps (doit correspondre aux poids SSL MONAI)
feature_size = 48
# Active le gradient checkpointing pour économiser la VRAM
use_checkpoint = True
# Chemin vers les poids SSL pré-entraînés du SwinViT
pretrained_path = "utils/model_swinvit.pt"
# Nombre de canaux du bottleneck SwinUNETR (feature_size=48 → 768)
bottleneck_channels = 768

# Dimension du token pour la cross-attention et la tête survie
d_model = 256
# Nombre de têtes de l'attention multi-têtes
n_heads = 4
# Dimension du vecteur clinique en entrée du MLP (écrasée par le ClinicalEncoder
# one-hot au moment de construire les loaders ; valeur ici n'est qu'un défaut)
n_clinical_features = 22
# Dimension cachée des têtes T/N après le GAP
hidden_tn = 256

# Nombre d'intervalles temporels pour la survie discrète
n_time_bins = 10
# Dimension cachée de la tête survie
surv_hidden = 256

# Taille de batch pour l'entraînement
batch_size = 2
# Learning rate initial pour AdamW
learning_rate = 1e-4
# Régularisation L2 pour AdamW
weight_decay = 1e-5
# Nombre total d'epochs d'entraînement
num_epochs = 350
# Nombre d'epochs de warm-up (seule la segmentation est entraînée)
n_warmup = 20
# Norme maximale pour le gradient clipping
grad_clip_norm = 1.0
# Puissance du scheduler PolynomialLR
poly_lr_power = 0.9

# --- Phase 2 : entraînement découplé des têtes cliniques sur features cachées ---
# Taille de batch (gros, car on n'opère que sur des bottlenecks pré-calculés)
clinical_batch_size = 64
# Nombre d'epochs pour la phase têtes cliniques
clinical_epochs = 200
# Learning rate pour les têtes (plus élevé : petits modules, pas de backbone)
clinical_lr = 1e-3

# Active les augmentations aléatoires à l'entraînement
use_augmentation = True
# Probabilité d'application de chaque augmentation
aug_probability = 0.5

# Fraction des données réservée à la validation
val_split = 0.2
# Graine aléatoire pour la reproductibilité
seed = 42

# Nombre de workers pour le DataLoader
num_workers = 4
# Fraction du dataset mise en cache en RAM par MONAI CacheDataset
cache_rate = 0.25
# Fréquence de sauvegarde des checkpoints intermédiaires (en epochs)
save_checkpoint_every = 5
# Active l'écriture des métriques dans TensorBoard
use_tensorboard = True

# Nom de l'expérience (crée un sous-dossier dédié)
experiment_name = "multitask_e2e"
# Dossier racine de sortie pour les expériences
output_dir = "experiments"
