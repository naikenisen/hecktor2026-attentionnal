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
from monai.inferers import sliding_window_inference


def train_one_epoch(model, loader, optimizer, scaler, device):
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
def validate(model, loader, device):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_label = AsDiscrete(to_onehot=config.num_seg_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=config.num_seg_classes)
    for batch in tqdm(loader, desc="Val", leave=False):
        ct_pet = batch["image"].to(device)
        seg_gt = batch["label"].to(device)
        with autocast():
            seg_logits = sliding_window_inference(
                ct_pet, roi_size=config.spatial_size, sw_batch_size=config.batch_size,
                predictor=lambda x: model(x)[0], overlap=0.5, mode="gaussian",
            )
        gt = [post_label(x) for x in decollate_batch(seg_gt)]
        pr = [post_pred(x) for x in decollate_batch(seg_logits)]
        dice_metric(y_pred=pr, y=gt)
    per_class = dice_metric.aggregate()        # un Dice par classe foreground (GTVp, GTVn)
    dice_metric.reset()
    return torch.nanmean(per_class).item(), per_class.tolist()


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    for d in (config.experiment_dir, config.checkpoint_dir):
        os.makedirs(d, exist_ok=True)

    # Récupère le meilleur LR du study Optuna
    storage = JournalStorage(JournalFileStorage(
        os.path.join(config.experiment_dir, "optuna_seg.log")))
    study = optuna.load_study(study_name="seg_swinunetr", storage=storage)
    lr = study.best_params["learning_rate"]
    print(f"retraining with best lr {lr:.6f} (best search dice {study.best_value:.4f})")

    train_loader, val_loader = PetCtDataModule(config).segmentation_loaders()

    model = SwinUNETRBackbone(
        input_channels=config.input_channels,
        num_classes=config.num_seg_classes,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        pretrained_path=config.pretrained_path,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config.seg_final_epochs, power=0.9)
    scaler = GradScaler()

    best_dice = 0.0
    for epoch in range(config.seg_final_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        dice, per_class = validate(model, val_loader, device)
        pc = "/".join(f"{d:.4f}" for d in per_class)
        print(f"epoch {epoch+1}/{config.seg_final_epochs} loss {train_loss:.4f} dice {dice:.4f} (GTVp/GTVn {pc})")
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), config.best_seg_path)
            print(f"saved best model (dice {dice:.4f})")
        scheduler.step()

    print(f"retrain complete, best dice {best_dice:.4f}")


if __name__ == "__main__":
    main()
