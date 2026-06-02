#!/usr/bin/env python3

import os
import torch
from tqdm import tqdm
import config
from src.networks import SwinUNETRBackbone
from src.dataset import get_feature_extraction_loaders


# Passe tout un split dans le backbone figé et collecte uniquement les bottlenecks.
# Les case_id permettent à la phase 2 de joindre les cibles cliniques (depuis le CSV).
@torch.no_grad()
def _extract_split(model, loader, device):
    feats, ids = [], []
    for batch in tqdm(loader, desc="Extract", leave=False):
        ct_pet = batch["image"].to(device, non_blocking=True)
        _, bottleneck = model(ct_pet)
        feats.append(bottleneck.float().cpu())
        ids.extend(batch["case_id"])
    return {
        "bottleneck": torch.cat(feats),
        "case_id":    ids,
    }


# Extrait et sauvegarde les bottlenecks des deux splits
@torch.no_grad()
def extract_features(model, config, device):
    model.eval()
    # Loaders déterministes (transforms de validation, sans shuffle)
    train_loader, val_loader = get_feature_extraction_loaders(config)
    os.makedirs(config.features_dir, exist_ok=True)

    print("extracting train split")
    torch.save(_extract_split(model, train_loader, device), config.train_features_path)
    print("extracting val split")
    torch.save(_extract_split(model, val_loader, device), config.val_features_path)
    print(f"features saved to {config.features_dir}")


def main():
    device = torch.device("cuda")

    model = SwinUNETRBackbone(
        input_channels=config.input_channels,
        num_classes=config.num_seg_classes,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        pretrained_path=config.pretrained_path,
    ).to(device)

    # Extraction des bottlenecks depuis le meilleur modèle de segmentation
    print("extracting bottlenecks")
    model.load_state_dict(torch.load(config.best_seg_path, map_location=device))
    extract_features(model, config, device)


if __name__ == "__main__":
    main()
