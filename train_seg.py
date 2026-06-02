import os
import torch
import torch.optim as optim
import optuna
from tqdm import tqdm
import config
from src.networks import SwinUNETRBackbone
from src.image_data import PetCtDataModule
from utils.losses import seg_loss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


def train_one_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        ct_pet = batch["image"].to(device, non_blocking=True)
        seg_gt = batch["label"].to(device, non_blocking=True)
        seg_logits, _ = model(ct_pet)
        loss = seg_loss(seg_logits, seg_gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.seg_grad_clip_norm)
        optimizer.step()
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


def objective(trial, train_loader, val_loader, device, global_best):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    poly_power = trial.suggest_float("poly_lr_power", 0.5, 1.5)

    model = build_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config.seg_epochs, power=poly_power)

    trial_best_dice = 0.0
    for epoch in range(config.seg_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
        dice = validate(model, val_loader, device, config)
        print(f"trial {trial.number} epoch {epoch+1} loss {train_loss:.4f} dice {dice:.4f}")

        if dice > trial_best_dice:
            trial_best_dice = dice
        if dice > global_best["dice"]:
            global_best["dice"] = dice
            torch.save(model.state_dict(), config.best_seg_path)
            print(f"saved new global best model (dice {dice:.4f})")

        trial.report(dice, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        scheduler.step()

    return trial_best_dice


def main():
    device = torch.device("cuda")
    for d in (config.experiment_dir, config.checkpoint_dir):
        os.makedirs(d, exist_ok=True)
    train_loader, val_loader = PetCtDataModule(config).segmentation_loaders()

    global_best = {"dice": 0.0}
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=config.seg_prune_warmup_epochs)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device, global_best),
        n_trials=config.seg_n_trials,
    )

    print(f"best dice {study.best_value:.4f}")
    print(f"best params {study.best_params}")


if __name__ == "__main__":
    main()
