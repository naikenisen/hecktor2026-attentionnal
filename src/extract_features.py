import os
import torch
from tqdm import tqdm
from src.dataset import get_feature_extraction_loaders


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


@torch.no_grad()
def extract_features(model, config, device):
    model.eval()
    train_loader, val_loader, train_df, _ = get_feature_extraction_loaders(config)
    out_dir = os.path.join(config.experiment_dir, "features")
    os.makedirs(out_dir, exist_ok=True)

    print("extracting train split")
    torch.save(_extract_split(model, train_loader, device),
               os.path.join(out_dir, "train.pt"))
    print("extracting val split")
    torch.save(_extract_split(model, val_loader, device),
               os.path.join(out_dir, "val.pt"))
    train_df.to_csv(os.path.join(out_dir, "train_df.csv"), index=False)
    print(f"features saved to {out_dir}")
