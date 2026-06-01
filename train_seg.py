#!/usr/bin/env python3

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import config
from src.swinunetr import SwinUNETRBackbone
from src.dataset import get_multitask_dataloaders, get_feature_extraction_loaders
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


# Passe tout un split dans le backbone figé et collecte bottlenecks + cibles tabulaires
@torch.no_grad()
def _extract_split(model, loader, device):
    feats, clin, t, n, time, event, ids = [], [], [], [], [], [], []
    for batch in tqdm(loader, desc="Extract", leave=False):
        ct_pet = batch["image"].to(device, non_blocking=True)
        _, bottleneck = model(ct_pet)
        feats.append(bottleneck.float().cpu())
        clin.append(batch["clinical"].float())
        t.append(batch["t_label"])
        n.append(batch["n_label"])
        time.append(batch["time"])
        event.append(batch["event"])
        ids.extend(batch["case_id"])
    return {
        "bottleneck": torch.cat(feats),
        "clinical":   torch.cat(clin),
        "t_label":    torch.cat(t),
        "n_label":    torch.cat(n),
        "time":       torch.cat(time),
        "event":      torch.cat(event),
        "case_id":    ids,
    }


# Extrait et sauvegarde les bottlenecks des deux splits (appelé après l'entraînement seg)
@torch.no_grad()
def extract_features(model, config, device):
    model.eval()
    # Loaders déterministes (transforms de validation, sans shuffle)
    train_loader, val_loader, train_df, _ = get_feature_extraction_loaders(config)
    out_dir = os.path.join(config.experiment_dir, "features")
    os.makedirs(out_dir, exist_ok=True)

    print("extracting train split")
    torch.save(_extract_split(model, train_loader, device),
               os.path.join(out_dir, "train.pt"))
    print("extracting val split")
    torch.save(_extract_split(model, val_loader, device),
               os.path.join(out_dir, "val.pt"))
    # train_df pour le calcul des bins temporels en phase 2 (train_clinical.py)
    train_df.to_csv(os.path.join(out_dir, "train_df.csv"), index=False)
    print(f"features saved to {out_dir}")


def main():
    device = torch.device("cuda")

    # Dossiers de sortie dérivés de la config
    config.experiment_dir = os.path.join(config.output_dir, config.experiment_name)
    config.checkpoint_dir = os.path.join(config.experiment_dir, "checkpoints")
    for d in (config.experiment_dir, config.checkpoint_dir):
        os.makedirs(d, exist_ok=True)

    # Chemin du meilleur modèle de segmentation
    best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    model = SwinUNETRBackbone(
        input_channels=config.input_channels,
        num_classes=config.num_seg_classes,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        pretrained_path=config.pretrained_path,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"segmentation backbone has {n_params:,} parameters")

    # DataLoaders (les champs tabulaires sont présents mais ignorés ici)
    train_loader, val_loader, _train_df, _clin = get_multitask_dataloaders(config)

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
            torch.save(model.state_dict(), best_path)
            print(f"saved best model")
        scheduler.step()

    # Extraction automatique des features depuis le meilleur modèle de segmentation
    print("extracting bottlenecks from best_model.pth")
    model.load_state_dict(torch.load(best_path, map_location=device))
    extract_features(model, config, device)


if __name__ == "__main__":
    main()
