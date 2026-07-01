# === Données sources (dataset HECKTOR sur le scratch Jean-Zay) ===
dataset_dir = "/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/dataset"
data_root = f"{dataset_dir}/hecktor_dataset_suv"
csv_path = f"{dataset_dir}/HECKTOR 2026 Training Data/HECKTOR_2026_training_data.csv"
csv_path_local = csv_path

# === Sorties d'un run : tout dans `results/` ===
# `results/` regroupe TOUT ce que produit un run (embeddings figés + modèle nnU-Net) et est
# destiné à être archivé puis supprimé après chaque exécution. Aucun dossier d'expérience,
# aucun versionnage : un seul chemin possible par artefact, écrasé à chaque run. Les chemins
# sont donc écrits en dur (pas d'arborescence reconstruite). Seule exception : le sous-dossier
# `nnUNet_results` (structure interne imposée par le paquet nnU-Net).
results_dir = "results"
train_features_path = "results/train.pt"      # embeddings figés du bottleneck nnU-Net (split train)
val_features_path = "results/val.pt"          # embeddings figés du bottleneck nnU-Net (split val)
datalist_path = "results/datalist.json"       # trace du split fourni à nnU-Net

# Le CSV clinique nettoyé (dropna) est rangé avec les données sources, pas dans results.
clinical_clean_csv_path = f"{dataset_dir}/clinical_clean.csv"

# === Segmentation nnU-Net (phase 1) ===
nnunet_dataset_id = 1                          # → Dataset001_HECKTOR
nnunet_dataset_name = "HECKTOR"
nnunet_dataset_dir = "Dataset001_HECKTOR"      # nom de dossier imposé par nnU-Net : Dataset{id:03d}_{name}
nnunet_modality = ["CT", "PET"]                # ordre des canaux : 0 = CT, 1 = PET (SUV)
nnunet_configuration = "3d_fullres"            # config nnU-Net entraînée
nnunet_trainer = "nnUNetTrainer"               # ex. "nnUNetTrainer_250epochs" pour limiter les époques
nnunet_plans = "nnUNetPlans"                   # nom des plans (défaut de l'experiment planner)
nnunet_checkpoint = "checkpoint_final.pth"     # checkpoint (dans results/nnUNet_results) rechargé pour l'extraction d'embeddings
nnunet_fold = 0
nnunet_use_project_split = True                # fold 0 = split_case_ids (cohérent avec le reste de la pipeline)
nnunet_gpu_memory_gb = 8                        # cible mémoire de l'experiment planner

# Arborescence nnU-Net. `nnUNet_raw` (liens symboliques, quasi vide) et `nnUNet_preprocessed`
# (lourd) restent sur le scratch, à côté des données ; seul le modèle entraîné (`nnUNet_results`)
# va dans `results/` — c'est l'unique sous-dossier de results, imposé par nnU-Net.
nnunet_raw_dir = f"{dataset_dir}/nnUNet_raw"
nnunet_preprocessed_dir = f"{dataset_dir}/nnUNet_preprocessed"
nnunet_results_dir = "results/nnUNet_results"
# Chemin du modèle entraîné. `{trainer}__{plans}__{config}` est la convention de nommage imposée
# par nnU-Net : reconstruite à partir des réglages ci-dessus pour retrouver checkpoints/summary.
nnunet_model_dir = f"{nnunet_results_dir}/{nnunet_dataset_dir}/{nnunet_trainer}__{nnunet_plans}__{nnunet_configuration}"

# === Têtes-forêts sur embedding figé (phase 2) ===
# Aucun hyperparamètre nnU-Net à régler (auto-configurant). Les seuls modèles ajustés sont les
# forêts (RandomForest T/N, RandomSurvivalForest survie), par GridSearchCV (grilles définies
# dans `tn.forest` et `src.survival_forest`). Aucun modèle de forêt n'est sauvegardé.
rf_seed = 42          # graine partagée RandomForest (T/N) et RandomSurvivalForest (survie)
val_split = 0.2       # proportion du split de validation
seed = 42             # graine du split train/val déterministe
num_workers = 4
