#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config import MultitaskConfig
from src.swinunetr import SwinUNETRConfig, SwinUNETRBackbone
from src.dataset import get_multitask_dataloaders, get_feature_extraction_loaders
from utils.losses import seg_loss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


# Instancie le backbone SwinUNETR (seg + bottleneck) depuis la config globale
def build_backbone(config, device):
    sw_cfg = SwinUNETRConfig(
        input_channels=config.input_channels,
        num_classes=config.num_seg_classes,
        spatial_size=config.spatial_size,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        pretrained_path=config.pretrained_path,
    )
    return SwinUNETRBackbone(sw_cfg).to(device)


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

        if not torch.isfinite(loss):
            print("[WARN] perte non finie - batch ignore")
            optimizer.zero_grad(set_to_none=True)
            continue

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


# Sauvegarde le backbone et l'état de l'optimiseur
def save_checkpoint(model, path, epoch, optimizer_state=None, **kwargs):
    ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), **kwargs}
    if optimizer_state is not None:
        ckpt["optimizer_state_dict"] = optimizer_state
    torch.save(ckpt, path)


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

    print("[Extract] train split...")
    torch.save(_extract_split(model, train_loader, device),
               os.path.join(out_dir, "train.pt"))
    print("[Extract] val split...")
    torch.save(_extract_split(model, val_loader, device),
               os.path.join(out_dir, "val.pt"))
    # train_df pour le calcul des bins temporels en phase 2 (train_clinical.py)
    train_df.to_csv(os.path.join(out_dir, "train_df.csv"), index=False)
    print(f"[Done] features -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--cuda-device", type=int, default=0)
    # Saute l'entraînement et n'extrait que les features depuis best_model.pth
    p.add_argument("--extract-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config = MultitaskConfig()

    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.cuda.set_device(device)
        print(f"[Device] {device} - {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print(f"[Device] {device}")

    # Chemin du meilleur modèle de segmentation
    best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    model = build_backbone(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] backbone segmentation : {n_params:,} parametres")

    # Mode extraction seule : recharge le best et dumpe les features
    if args.extract_only:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Load] {best_path}")
        extract_features(model, config, device)
        return

    # DataLoaders (les champs tabulaires sont présents mais ignorés ici)
    train_loader, val_loader, _train_df, _clin = get_multitask_dataloaders(config)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config.num_epochs, power=config.poly_lr_power)

    writer = SummaryWriter(config.log_dir) if config.use_tensorboard else None

    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_dice = ckpt.get("best_metric", 0.0)
        print(f"[Resume] epoch={start_epoch}  best_dice={best_dice:.4f}")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
        print(f"Train loss : {train_loss:.4f}")
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)

        should_val = (epoch + 1) % 5 == 0 or (epoch + 1) == config.num_epochs
        if should_val:
            dice = validate(model, val_loader, device, config)
            print(f"Val   : Dice={dice:.4f}")
            if writer:
                writer.add_scalar("Val/dice", dice, epoch)
            if dice > best_dice:
                best_dice = dice
                save_checkpoint(model, best_path, epoch, optimizer.state_dict(),
                                best_metric=best_dice)
                print(f"[Save] new best model (Dice={dice:.4f}) -> {best_path}")

        scheduler.step()
        if writer:
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if config.save_checkpoint_every > 0 and (epoch + 1) % config.save_checkpoint_every == 0:
            save_checkpoint(model, os.path.join(config.checkpoint_dir, "last_model.pth"),
                            epoch, optimizer.state_dict(), best_metric=best_dice)

    if writer:
        writer.close()
    print("\nSegmentation training complete.")

    # Extraction automatique des features depuis le meilleur modèle de segmentation
    print("\n[Pipeline] extraction des bottlenecks depuis best_model.pth...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    extract_features(model, config, device)


if __name__ == "__main__":
    main()
