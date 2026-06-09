import os
data_root = "/work/imvia/in156281/datasets/hecktor_dataset_preprocessed_suv"
csv_path = "/work/imvia/in156281/datasets/hecktor_dataset/HECKTOR_2026_training_data.csv"
input_channels = 2
num_seg_classes = 3
spatial_size = (128, 128, 128)
feature_size = 48
use_checkpoint = True
pretrained_path = "utils/model_swinvit.pt"
batch_size = 2

# Segmentation training
seg_search_epochs = 50          # epochs par trial pendant la HP search
seg_final_epochs = 200          # epochs du retrain final avec les meilleurs HP
seg_grad_clip_norm = 1.0
seg_n_trials = 20               # trials par worker (4 workers = 80 trials totaux)
seg_search_timeout_hours = 6    # mur de temps de la HP search par worker

# Têtes tabulaires sur embedding figé (forêts aléatoires, aucune fusion)
rf_seed = 42
tn_n_trials = 50      # trials Optuna par RandomForest T / N
surv_n_trials = 30    # trials Optuna du RandomSurvivalForest
use_augmentation = True
aug_probability = 0.5
val_split = 0.2
seed = 42
num_workers = 4
cache_rate = 0.25
experiment_name = "multitask_e2e"
output_dir = "experiments"
experiment_dir = os.path.join(output_dir, experiment_name)
checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
features_dir = os.path.join(experiment_dir, "features")
best_seg_path = os.path.join(checkpoint_dir, "best_model.pth")
best_tn_t_path = os.path.join(checkpoint_dir, "tn_t_rf.joblib")
best_tn_n_path = os.path.join(checkpoint_dir, "tn_n_rf.joblib")
best_survival_path = os.path.join(checkpoint_dir, "survival_rsf.joblib")
train_features_path = os.path.join(features_dir, "train.pt")
val_features_path = os.path.join(features_dir, "val.pt")

# Survie sur embedding CT du modèle de fondation CT-FM (SegResNet SSL, 512-D, CT seule)
foundation_model_id = "project-lighter/ct_fm_feature_extractor"
foundation_train_features_path = os.path.join(features_dir, "ct_fm_train.pt")
foundation_val_features_path = os.path.join(features_dir, "ct_fm_val.pt")
best_foundation_survival_path = os.path.join(checkpoint_dir, "survival_ct_fm_rsf.joblib")
