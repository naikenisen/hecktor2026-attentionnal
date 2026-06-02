import os
import torch
import torch.optim as optim
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import config
from src.networks import SwinUNETRBackbone
from src.image_data import PetCtDataModule
from utils.losses import seg_loss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


def train_one_epoch(model, loader, optimizer, scaler, device, config):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        ct_pet = batch["image"].to(device, non_blocking=True)
        seg_gt = batch["label"].to(device, non_blocking=True)
        with autocast():
            seg_logits, _ = model(ct_pet)
            loss = seg_loss(seg_logits, seg_gt)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.seg_grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_label = AsDiscrete(to_onehot=config.num_seg_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=config.num_seg_classes)
    for batch in tqdm(loader, desc="Val", leave=False):
        ct_pet = batch["image"].to(device)
        seg_gt = batch["label"].to(device)
        with autocast():
            seg_logits, _ = model(ct_pet)
        gt = [post_label(x) for x in decollate_batch(seg_gt)]
        pr = [post_pred(x) for x in decollate_batch(seg_logits)]
        dice_metric(y_pred=pr, y=gt)
    dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return dice


def build_model(device):
    return SwinUNETRBackbone(
        input_channels=config.input_channels,
        num_classes=config.num_seg_classes,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        pretrained_path=config.pretrained_path,
    ).to(device)


def objective(trial, train_loader, val_loader, device):
    # Seul le LR est tuné — wd et poly_power ont des optimums plats sur ce type de tâche
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = 1e-5
    poly_power = 0.9

    model = build_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config.seg_search_epochs, power=poly_power)
    scaler = GradScaler()

    trial_best_dice = 0.0
    try:
        for epoch in range(config.seg_search_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, config)
            dice = validate(model, val_loader, device, config)
            print(f"trial {trial.number} epoch {epoch+1} loss {train_loss:.4f} dice {dice:.4f}")

            if dice > trial_best_dice:
                trial_best_dice = dice

            trial.report(dice, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            scheduler.step()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise optuna.TrialPruned()

    return trial_best_dice


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    for d in (config.experiment_dir, config.checkpoint_dir):
        os.makedirs(d, exist_ok=True)
    train_loader, val_loader = PetCtDataModule(config).segmentation_loaders()

    storage = JournalStorage(JournalFileStorage(
        os.path.join(config.experiment_dir, "optuna_seg.log")))
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5,
        max_resource=config.seg_search_epochs,
        reduction_factor=3,
    )
    study = optuna.create_study(
        storage=storage,
        study_name="seg_swinunetr",
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device),
        n_trials=config.seg_n_trials,
        timeout=config.seg_search_timeout_hours * 3600,
        catch=(RuntimeError,),
    )

    print(f"best dice {study.best_value:.4f}")
    print(f"best params {study.best_params}")


if __name__ == "__main__":
    main()
