#!/usr/bin/env python3

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import config
from src.swinunetr import SwinUNETRBackbone
from src.dataset import get_seg_dataloaders
from utils.losses import seg_loss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


# Entraîne la segmentation sur un epoch complet et retourne la perte moyenne
def train_one_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        # Images bimodales CT+PET
        ct_pet = batch["image"].to(device, non_blocking=True)
        # Masque de segmentation de référence
        seg_gt = batch["label"].to(device, non_blocking=True)
        # Forward : on ne garde que les logits de segmentation
        seg_logits, _ = model(ct_pet)
        loss = seg_loss(seg_logits, seg_gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# Évalue la segmentation : Dice moyen (hors fond) sur le split de validation
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


def main():
    device = torch.device("cuda")

    # Crée les dossiers de sortie (chemins définis dans config.py)
    for d in (config.experiment_dir, config.checkpoint_dir):
        os.makedirs(d, exist_ok=True)

    model = SwinUNETRBackbone(
        input_channels=config.input_channels,
        num_classes=config.num_seg_classes,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        pretrained_path=config.pretrained_path,
    ).to(device)

    train_loader, val_loader = get_seg_dataloaders(config)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config.num_epochs, power=config.poly_lr_power)

    best_dice = 0.0
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
        dice = validate(model, val_loader, device, config)
        print(f"epoch : {epoch+1} loss : {train_loss:.4f} dice : {dice:.4f}")
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), config.best_seg_path)
            print(f"saved best model")
        scheduler.step()


if __name__ == "__main__":
    main()
