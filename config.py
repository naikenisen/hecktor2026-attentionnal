import os
data_root = "/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/dataset/hecktor_dataset_suv"
csv_path = "/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/dataset/HECKTOR 2026 Training Data/HECKTOR_2026_training_data.csv"
csv_path_local="/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/dataset/HECKTOR 2026 Training Data/HECKTOR_2026_training_data.csv"

# Segmentation nnU-Net (MONAI nnUNetV2Runner + paquet nnunetv2) — phase 1
nnunet_dataset_id = 1                          # → Dataset001_HECKTOR
nnunet_dataset_name = "HECKTOR"
nnunet_modality = ["CT", "PET"]                # ordre des canaux : 0 = CT, 1 = PET (SUV)
nnunet_configuration = "3d_fullres"            # config nnU-Net entraînée
nnunet_trainer = "nnUNetTrainer"               # ex. "nnUNetTrainer_250epochs" pour limiter
nnunet_plans = "nnUNetPlans"                    # nom des plans (défaut de l'experiment planner)
nnunet_checkpoint = "checkpoint_final.pth"      # checkpoint rechargé pour l'extraction d'embeddings
nnunet_fold = 0
nnunet_use_project_split = True                # fold 0 = split_case_ids (cohérent pipeline)
nnunet_gpu_memory_gb = 8                        # cible mémoire de l'experiment planner

# Têtes tabulaires sur embedding figé (forêts aléatoires, aucune fusion)
rf_seed = 42
tn_n_trials = 50      # trials Optuna par RandomForest T / N
surv_n_trials = 30    # itérations RandomizedSearchCV (CV K-fold) du RandomSurvivalForest
val_split = 0.2
seed = 42
num_workers = 4
experiment_name = "multitask_e2e"
output_dir = "experiments"
experiment_dir = os.path.join(output_dir, experiment_name)
checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
features_dir = os.path.join(experiment_dir, "features")

# Version des artefacts de phase 2 (embeddings figés + têtes tabulaires/survie). Incrémenter
# pour tout régénérer sur la population COURANTE sans écraser les anciens embeddings ni les
# anciens résultats : chaque fichier porte le suffixe `_<features_version>`. (Le modèle de
# segmentation nnU-Net, phase 1, n'est PAS versionné — il reste réutilisé tel quel.)
features_version = "v2"


def _versioned(path: str) -> str:
    """Insère le suffixe de version avant l'extension : `foo.pt` → `foo_v2.pt`."""
    root, ext = os.path.splitext(path)
    return f"{root}_{features_version}{ext}"


best_tn_t_path = _versioned(os.path.join(checkpoint_dir, "tn_t_rf.joblib"))
best_tn_n_path = _versioned(os.path.join(checkpoint_dir, "tn_n_rf.joblib"))
best_survival_path = _versioned(os.path.join(checkpoint_dir, "survival_rsf.joblib"))
best_survival_tn_path = _versioned(os.path.join(checkpoint_dir, "survival_rsf_with_tn.joblib"))
clinical_clean_csv_path = _versioned(os.path.join(experiment_dir, "clinical_clean.csv"))
best_nnunet_survival_path = _versioned(os.path.join(checkpoint_dir, "survival_nnunet_rsf.joblib"))
train_features_path = _versioned(os.path.join(features_dir, "train.pt"))
val_features_path = _versioned(os.path.join(features_dir, "val.pt"))

# Arborescence nnU-Net (raw / preprocessed / results) sous l'expérience
nnunet_work_dir = os.path.join(experiment_dir, "nnunet")
nnunet_raw_dir = os.path.join(nnunet_work_dir, "nnUNet_raw")
nnunet_preprocessed_dir = os.path.join(nnunet_work_dir, "nnUNet_preprocessed")
nnunet_results_dir = os.path.join(nnunet_work_dir, "nnUNet_results")
nnunet_dataset_dir = f"Dataset{nnunet_dataset_id:03d}_{nnunet_dataset_name}"
nnunet_model_dir = os.path.join(
    nnunet_results_dir, nnunet_dataset_dir,
    f"{nnunet_trainer}__{nnunet_plans}__{nnunet_configuration}")

# Survie sur embedding CT du modèle de fondation CT-FM (SegResNet SSL, 512-D, CT seule)
foundation_model_id = "project-lighter/ct_fm_feature_extractor"
foundation_train_features_path = _versioned(os.path.join(features_dir, "ct_fm_train.pt"))
foundation_val_features_path = _versioned(os.path.join(features_dir, "ct_fm_val.pt"))
best_foundation_survival_path = _versioned(os.path.join(checkpoint_dir, "survival_ct_fm_rsf.joblib"))
