#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import config
from src.clinical_model import ClinicalModel
from utils.losses import DeepHitDiscreteLoss, UncertaintyWeightedLoss
from utils.metrics import balanced_accuracy, discrete_risk_from_surv_logits, c_index


# Calcule les bornes des bins de survie par quantiles sur les temps d'événement du train
def compute_bin_edges(train_df, n_bins: int) -> np.ndarray:
    # Temps des patients avec un événement observé (rechute)
    ev = train_df[train_df["Relapse"] == 1]["RFS"].dropna().values.astype(float)
    if len(ev) < n_bins:
        # Fallback sur tous les temps si trop peu d'événements
        ev = train_df["RFS"].dropna().values.astype(float)
    if len(ev) == 0:
        return np.linspace(0, 1, n_bins + 1)
    # Quantiles équiprobables définissant les bornes des bins
    edges = np.quantile(ev, np.linspace(0, 1, n_bins + 1))
    # Bornes infinies pour capturer tous les temps possibles
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


# Charge un split de features pré-calculées et déplace les tenseurs cibles sur le device
def load_features(path: str, device) -> dict:
    d = torch.load(path, map_location="cpu")
    # Le bottleneck reste sur CPU (volumineux) : on l'indexe par batch au moment voulu
    return d


# Poids de classe inverses de fréquence (les labels −1 inconnus sont ignorés)
def class_weights(labels: torch.Tensor, num_classes: int, device) -> torch.Tensor:
    valid = labels[labels >= 0]
    counts = torch.bincount(valid, minlength=num_classes).float().clamp_min(1.0)
    # w_c = N / (K * n_c) — somme pondérée équilibrée entre classes
    w = counts.sum() / (num_classes * counts)
    return w.to(device)


# Entraîne les têtes sur un epoch complet à partir des features cachées
def train_one_epoch(model, data, idx_perm, optimizer, weighting, deephit,
                    ce_t, ce_n, bin_edges, device, config):
    model.train()
    total_loss = 0.0
    n_batches = 0
    bs = config.clinical_batch_size

    for i in range(0, len(idx_perm), bs):
        idx = idx_perm[i:i + bs]
        # Bottleneck du batch (déplacé sur GPU à la volée)
        bn = data["bottleneck"][idx].to(device, non_blocking=True)
        clin = data["clinical"][idx].to(device, non_blocking=True)
        t_lbl = data["t_label"][idx].to(device)
        n_lbl = data["n_label"][idx].to(device)
        times = data["time"][idx].to(device)
        events = data["event"][idx].to(device)

        out = model(bn, clin)
        l_t = ce_t(out["t_logits"], t_lbl)
        l_n = ce_n(out["n_logits"], n_lbl)

        # Survie : on n'utilise que les patients avec un temps valide (RFS connu > 0)
        valid = times > 0
        if valid.any():
            l_srv = deephit(out["surv_logits"][valid], times[valid],
                            events[valid], bin_edges)
        else:
            l_srv = torch.tensor(0.0, device=device)

        # Pondération automatique par incertitude sur les 3 tâches
        loss, _ = weighting(l_t, l_n, l_srv)

        if not torch.isfinite(loss):
            print("[WARN] perte non finie - batch ignore")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(weighting.parameters()),
            max_norm=config.grad_clip_norm,
        )
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# Évalue les têtes sur le split de validation (BalAcc T/N, C-index)
@torch.no_grad()
def validate(model, data, bin_edges, device, config):
    model.eval()
    t_true, t_pred = [], []
    n_true, n_pred = [], []
    risks, all_t, all_e = [], [], []
    bs = config.clinical_batch_size
    N = data["bottleneck"].size(0)

    for i in range(0, N, bs):
        idx = slice(i, i + bs)
        bn = data["bottleneck"][idx].to(device)
        clin = data["clinical"][idx].to(device)
        out = model(bn, clin)

        t_true.extend(data["t_label"][idx].numpy().tolist())
        t_pred.extend(out["t_logits"].argmax(1).cpu().numpy().tolist())
        n_true.extend(data["n_label"][idx].numpy().tolist())
        n_pred.extend(out["n_logits"].argmax(1).cpu().numpy().tolist())

        r = discrete_risk_from_surv_logits(out["surv_logits"]).cpu().numpy()
        risks.extend(r.tolist())
        all_t.extend(data["time"][idx].numpy().tolist())
        all_e.extend(data["event"][idx].numpy().tolist())

    risks, all_t, all_e = np.array(risks), np.array(all_t), np.array(all_e)
    # C-index calculé uniquement sur les patients à temps valide
    valid = all_t > 0
    ci = c_index(risks[valid], all_t[valid], all_e[valid]) if valid.any() else 0.5

    return {
        "bal_t":  balanced_accuracy(np.array(t_true), np.array(t_pred)),
        "bal_n":  balanced_accuracy(np.array(n_true), np.array(n_pred)),
        "c_index": ci,
    }


def main():
    device = torch.device("cuda")
    print(f"[Device] {device} - {torch.cuda.get_device_name(device)}")

    # Dossiers de sortie dérivés de la config
    config.experiment_dir = os.path.join(config.output_dir, config.experiment_name)
    config.checkpoint_dir = os.path.join(config.experiment_dir, "checkpoints")
    config.log_dir = os.path.join(config.experiment_dir, "logs")
    for d in (config.experiment_dir, config.checkpoint_dir, config.log_dir):
        os.makedirs(d, exist_ok=True)

    # Features pré-calculées
    feat_dir = os.path.join(config.experiment_dir, "features")
    train = load_features(os.path.join(feat_dir, "train.pt"), device)
    val = load_features(os.path.join(feat_dir, "val.pt"), device)
    train_df = pd.read_csv(os.path.join(feat_dir, "train_df.csv"))
    print(f"[Data] {train['bottleneck'].size(0)} train / {val['bottleneck'].size(0)} val features")

    # Aligne la dim du ClinicalMLP sur les vecteurs cliniques cachés (one-hot)
    config.n_clinical_features = train["clinical"].size(1)
    print(f"[Data] dim features cliniques = {config.n_clinical_features}")

    # Bins temporels (quantiles sur les événements du train)
    bin_edges = torch.tensor(
        compute_bin_edges(train_df, config.n_time_bins),
        dtype=torch.float32, device=device,
    )

    model = ClinicalModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] tetes cliniques : {n_params:,} parametres")

    # 3 tâches désormais : T, N, survie
    weighting = UncertaintyWeightedLoss(n_tasks=3).to(device)
    deephit = DeepHitDiscreteLoss(alpha=0.2).to(device)
    # CrossEntropy pondérées par l'inverse des fréquences de classe
    ce_t = nn.CrossEntropyLoss(ignore_index=-1,
                               weight=class_weights(train["t_label"], config.num_t_classes, device))
    ce_n = nn.CrossEntropyLoss(ignore_index=-1,
                               weight=class_weights(train["n_label"], config.num_n_classes, device))

    optimizer = optim.AdamW(
        list(model.parameters()) + list(weighting.parameters()),
        lr=config.clinical_lr, weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config.clinical_epochs, power=config.poly_lr_power,
    )

    writer = SummaryWriter(os.path.join(config.log_dir, "clinical")) \
        if config.use_tensorboard else None
    ckpt_dir = config.checkpoint_dir
    best_metric = 0.0
    N_train = train["bottleneck"].size(0)

    for epoch in range(config.clinical_epochs):
        # Permutation aléatoire des indices d'entraînement
        perm = torch.randperm(N_train)
        train_loss = train_one_epoch(
            model, train, perm, optimizer, weighting, deephit,
            ce_t, ce_n, bin_edges, device, config,
        )
        print(f"\n=== Epoch {epoch+1}/{config.clinical_epochs} ===")
        print(f"Train loss : {train_loss:.4f}")
        if writer:
            writer.add_scalar("ClinLoss/train", train_loss, epoch)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.clinical_epochs:
            metrics = validate(model, val, bin_edges, device, config)
            print(f"Val   : BalAcc T={metrics['bal_t']:.4f}  "
                  f"N={metrics['bal_n']:.4f}  C-index={metrics['c_index']:.4f}")
            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"ClinVal/{k}", v, epoch)

            # Métrique combinée : moyenne des 3 tâches cliniques
            combined = (metrics["bal_t"] + metrics["bal_n"] + metrics["c_index"]) / 3
            if combined > best_metric:
                best_metric = combined
                path = os.path.join(ckpt_dir, "best_clinical.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_metric": best_metric,
                }, path)
                print(f"[Save] new best clinical ({combined:.4f}) -> {path}")

        scheduler.step()
        if writer:
            writer.add_scalar("ClinLR", optimizer.param_groups[0]["lr"], epoch)

    if writer:
        writer.close()
    print("\nClinical training complete.")


if __name__ == "__main__":
    main()
