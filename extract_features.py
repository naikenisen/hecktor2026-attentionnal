#!/usr/bin/env python3

import os
import torch
from tqdm import tqdm
import config
from src.swinunetr import SwinUNETRBackbone
from src.dataset import get_feature_extraction_loaders


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


# Extrait et sauvegarde les bottlenecks des deux splits
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

    # Chemin du meilleur modèle de segmentation
    best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    model = SwinUNETRBackbone(
        input_channels=config.input_channels,
        num_classes=config.num_seg_classes,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        pretrained_path=config.pretrained_path,
    ).to(device)

    # Extraction des bottlenecks depuis le meilleur modèle de segmentation
    print("extracting bottlenecks from best_model.pth")
    model.load_state_dict(torch.load(best_path, map_location=device))
    extract_features(model, config, device)


if __name__ == "__main__":
    main()
