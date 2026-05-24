#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Ajoute la racine du projet au path pour les imports relatifs
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from config import MultitaskConfig
from src.model import MultitaskModel
from src.dataset import get_multitask_dataloaders
from utils.losses import seg_loss, t_loss, n_loss, DeepHitDiscreteLoss, UncertaintyWeightedLoss as UncertaintyWeighting
from utils.metrics import (
    balanced_accuracy,
    discrete_risk_from_surv_logits,
    c_index,
)
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


# Active ou désactive le calcul de gradient pour tous les paramètres d'un module
def freeze(module: torch.nn.Module, frozen: bool):
    for p in module.parameters():
        p.requires_grad = not frozen


# Calcule les bornes des T bins de survie par quantiles sur les temps d'événement du train
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
    # Borne inférieure à −∞ pour capturer tous les temps possibles
    edges[0] = -np.inf
    # Borne supérieure à +∞ pour capturer tous les temps possibles
    edges[-1] = np.inf
    return edges


# Entraîne le modèle sur un epoch complet et retourne la perte moyenne
def train_one_epoch(model, loader, optimizer, weighting, deephit, bin_edges,
                    device, config, warmup: bool):
    model.train()
    # Accumulateur de perte sur l'epoch
    total_loss = 0.0
    # Compteur de batches pour la moyenne
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        # Images bimodales CT+PET (B, 2, D, H, W)
        ct_pet = batch["image"].to(device, non_blocking=True)
        # Masque de segmentation de référence (B, 1, D, H, W)
        seg_gt = batch["label"].to(device, non_blocking=True)
        # Vecteur clinique (B, 7)
        clinical = batch["clinical"].to(device, non_blocking=True).float()
        # Labels de staging T (B,)
        t_lbl = batch["t_label"].to(device, non_blocking=True)
        # Labels de staging N (B,)
        n_lbl = batch["n_label"].to(device, non_blocking=True)
        # Temps de suivi RFS (B,)
        times = batch["time"].to(device, non_blocking=True)
        # Indicateurs d'événement (B,)
        events = batch["event"].to(device, non_blocking=True)

        # Forward complet du modèle multitâche
        out = model(ct_pet, clinical)

        # Perte de segmentation Dice+Focal
        l_seg = seg_loss(out["seg_mask"], seg_gt)
        # Perte de staging T
        l_t = t_loss(out["t_logits"], t_lbl)
        # Perte de staging N
        l_n = n_loss(out["n_logits"], n_lbl)
        # Perte de survie DeepHit discrète
        l_srv = deephit(out["surv_logits"], times, events, bin_edges)

        if warmup:
            # Phase warm-up : seule la segmentation est optimisée
            loss = l_seg
        else:
            # Phase normale : pondération automatique par incertitude
            loss, _ = weighting(l_seg, l_t, l_n, l_srv)

        # Garde anti-NaN : on saute le batch sans stepper si la perte explose,
        # pour ne pas corrompre les poids du modèle avec des gradients non finis
        if not torch.isfinite(loss):
            print("[WARN] perte non finie detectee - batch ignore")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping sur tous les paramètres optimisés (modèle + pondération)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(weighting.parameters()),
            max_norm=config.grad_clip_norm,
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# Évalue le modèle sur le loader de validation et retourne Dice, BalAcc T/N, C-index
@torch.no_grad()
def validate(model, loader, deephit, bin_edges, device, config):
    model.eval()
    # Métrique Dice MONAI agrégée sur tout le split de validation
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    # Transform one-hot pour les labels de référence
    post_label = AsDiscrete(to_onehot=config.num_seg_classes)
    # Transform argmax + one-hot pour les prédictions de segmentation
    post_pred = AsDiscrete(argmax=True, to_onehot=config.num_seg_classes)

    # Listes pour accumuler les prédictions et vérités de staging T
    t_true, t_pred = [], []
    # Listes pour accumuler les prédictions et vérités de staging N
    n_true, n_pred = [], []
    # Listes pour accumuler les scores de risque, temps et événements de survie
    risks, all_t, all_e = [], [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        # Images bimodales CT+PET
        ct_pet = batch["image"].to(device)
        # Masque de référence
        seg_gt = batch["label"].to(device)
        # Vecteur clinique
        clinical = batch["clinical"].to(device).float()
        # Forward sans gradient
        out = model(ct_pet, clinical)

        # Calcul du Dice via MONAI (one-hot sur labels et prédictions)
        gt_list = [post_label(x) for x in decollate_batch(seg_gt)]
        pr_list = [post_pred(x) for x in decollate_batch(out["seg_mask"])]
        dice_metric(y_pred=pr_list, y=gt_list)

        # Accumulation des labels et prédictions de staging T
        t_true.extend(batch["t_label"].numpy().tolist())
        t_pred.extend(out["t_logits"].argmax(1).cpu().numpy().tolist())
        # Accumulation des labels et prédictions de staging N
        n_true.extend(batch["n_label"].numpy().tolist())
        n_pred.extend(out["n_logits"].argmax(1).cpu().numpy().tolist())

        # Scores de risque scalaires dérivés des logits de survie
        r = discrete_risk_from_surv_logits(out["surv_logits"]).cpu().numpy()
        risks.extend(r.tolist())
        all_t.extend(batch["time"].numpy().tolist())
        all_e.extend(batch["event"].numpy().tolist())

    # Dice moyen sur le split de validation (hors fond)
    dice = dice_metric.aggregate().item()
    dice_metric.reset()

    # Balanced Accuracy du staging T
    bal_t = balanced_accuracy(np.array(t_true), np.array(t_pred))
    # Balanced Accuracy du staging N
    bal_n = balanced_accuracy(np.array(n_true), np.array(n_pred))
    # C-index de concordance pour la survie
    ci = c_index(np.array(risks), np.array(all_t), np.array(all_e))

    return {"dice": dice, "bal_t": bal_t, "bal_n": bal_n, "c_index": ci}


# Parse les arguments de la ligne de commande (resume, cuda-device)
def parse_args():
    # Parser argparse avec les deux arguments optionnels
    p = argparse.ArgumentParser()
    # Chemin vers un checkpoint pour reprendre l'entraînement
    p.add_argument("--resume", type=str, default=None)
    # Index GPU à utiliser
    p.add_argument("--cuda-device", type=int, default=0)
    return p.parse_args()


# Point d'entrée principal : initialise tout et lance la boucle d'entraînement
def main():
    # Arguments de la ligne de commande
    args = parse_args()
    # Configuration globale de la pipeline
    config = MultitaskConfig()

    if config.device == "cuda" and torch.cuda.is_available():
        # Périphérique GPU sélectionné
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.cuda.set_device(device)
        print(f"[Device] {device} - {torch.cuda.get_device_name(device)}")
    else:
        # Repli sur CPU si CUDA indisponible
        device = torch.device("cpu")
        print(f"[Device] {device}")

    # DataLoaders train/val, DataFrame train et encodeur clinique
    train_loader, val_loader, train_df, _clin_enc = get_multitask_dataloaders(config)
    # Bornes des bins temporels calculées sur les événements du train set (numpy)
    bin_edges_np = compute_bin_edges(train_df, config.n_time_bins)
    # Bornes converties en tensor sur le bon périphérique
    bin_edges = torch.tensor(bin_edges_np, dtype=torch.float32, device=device)
    print(f"[Data] bin edges (T={config.n_time_bins}) : {bin_edges_np}")

    # Modèle multitâche complet déplacé sur le périphérique cible
    model = MultitaskModel(config).to(device)
    # Nombre total de paramètres du modèle
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n_params:,} parametres")

    # Module de pondération par incertitude (ses paramètres sont optimisés conjointement)
    weighting = UncertaintyWeighting(n_tasks=4).to(device)
    # Instance de la perte DeepHit discrète
    deephit = DeepHitDiscreteLoss(alpha=0.2).to(device)

    # Optimiseur AdamW sur les paramètres du modèle et de la pondération
    optimizer = optim.AdamW(
        list(model.parameters()) + list(weighting.parameters()),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    # Scheduler PolynomialLR qui décroît le LR sur toute la durée d'entraînement
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config.num_epochs, power=config.poly_lr_power,
    )

    # Writer TensorBoard (None si désactivé dans la config)
    writer = SummaryWriter(config.log_dir) if config.use_tensorboard else None

    # Epoch de départ (0 sauf si reprise depuis un checkpoint)
    start_epoch = 0
    # Meilleure métrique combinée observée (pour la sélection du meilleur modèle)
    best_metric = 0.0
    if args.resume:
        # Chargement du checkpoint et restauration de l'état de l'optimiseur
        ckpt = model.load_checkpoint(args.resume, device)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"[Resume] epoch={start_epoch}  best={best_metric:.4f}")

    for epoch in range(start_epoch, config.num_epochs):
        # Indicateur de phase warm-up : True tant qu'on n'a pas atteint n_warmup epochs
        warmup = epoch < config.n_warmup

        # Gel ou dégel des têtes T/N/Survie et du bloc de fusion selon la phase
        for mod in [model.t_head, model.n_head, model.surv_head,
                    model.cross_attn, model.clin_mlp, model.proj_tn]:
            freeze(mod, frozen=warmup)
        # Le CLS token est également gelé pendant le warm-up
        model.cls_token.requires_grad = not warmup

        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} {'[warm-up seg]' if warmup else ''} ===")
        # Perte d'entraînement moyenne sur l'epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, weighting, deephit, bin_edges,
            device, config, warmup,
        )
        print(f"Train loss : {train_loss:.4f}")
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)

        # Validation tous les 5 epochs et à la dernière epoch
        should_val = (epoch + 1) % 5 == 0 or (epoch + 1) == config.num_epochs
        if should_val:
            # Dictionnaire des métriques de validation
            metrics = validate(model, val_loader, deephit, bin_edges, device, config)
            print(f"Val   : Dice={metrics['dice']:.4f}  "
                  f"BalAcc T={metrics['bal_t']:.4f}  N={metrics['bal_n']:.4f}  "
                  f"C-index={metrics['c_index']:.4f}")
            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"Val/{k}", v, epoch)

            # Métrique combinée pondérée : Dice compte double (tâche principale)
            combined = 0.4 * metrics["dice"] + 0.2 * metrics["bal_t"] \
                       + 0.2 * metrics["bal_n"] + 0.2 * metrics["c_index"]
            if combined > best_metric:
                best_metric = combined
                # Chemin du meilleur checkpoint
                path = os.path.join(config.checkpoint_dir, "best_model.pth")
                model.save_checkpoint(path, epoch, optimizer.state_dict(),
                                      best_metric=best_metric)
                print(f"[Save] new best model ({combined:.4f}) -> {path}")

        # Décroissance du LR après chaque epoch
        scheduler.step()
        # Learning rate courant après le step
        lr = optimizer.param_groups[0]["lr"]
        if writer:
            writer.add_scalar("LR", lr, epoch)

        if config.save_checkpoint_every > 0 and (epoch + 1) % config.save_checkpoint_every == 0:
            # Chemin du checkpoint périodique (dernier état)
            path = os.path.join(config.checkpoint_dir, "last_model.pth")
            model.save_checkpoint(path, epoch, optimizer.state_dict(),
                                  best_metric=best_metric)

    if writer:
        writer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
