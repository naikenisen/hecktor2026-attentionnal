import os
data_root = "/work/imvia/in156281/datasets/hecktor_dataset_preprocessed"
csv_path = "/work/imvia/in156281/datasets/hecktor_dataset/HECKTOR_2026_training_data.csv"
input_channels = 2
num_seg_classes = 3
num_t_classes = 4
num_n_classes = 3
spatial_size = (128, 128, 128)
feature_size = 48
use_checkpoint = True
pretrained_path = "utils/model_swinvit.pt"
bottleneck_channels = 768
d_model = 256
n_heads = 4
n_clinical_features = 22
hidden_tn = 256
n_time_bins = 10
surv_hidden = 256
batch_size = 2

# Segmentation training
seg_search_epochs = 50          # epochs par trial pendant la HP search
seg_final_epochs = 200          # epochs du retrain final avec les meilleurs HP
seg_grad_clip_norm = 1.0
seg_n_trials = 20               # trials par worker (4 workers = 80 trials totaux)
seg_search_timeout_hours = 6    # mur de temps de la HP search par worker

# Clinical training
clinical_batch_size = 64
clinical_epochs = 200
clinical_grad_clip_norm = 1.0
clinical_n_trials = 100
clinical_prune_warmup_epochs = 20
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
best_clinical_path = os.path.join(checkpoint_dir, "best_clinical.pth")
train_features_path = os.path.join(features_dir, "train.pt")
val_features_path = os.path.join(features_dir, "val.pt")
