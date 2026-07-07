"""Classifieur ResNet-18 3D : invasion T4 vs {T2, T3} sur la ROI CT de la tumeur.

Complément de `TN/T.py` : la taille (diamètre axial) sépare déjà T0/T1/T2/T3, mais **jamais**
T4 — le stade T4 est défini par l'*invasion de structures adjacentes* (os, cartilage, muscles),
pas par un seuil de taille. On apprend donc cette invasion directement sur l'image.

Entrée réseau (2 canaux, ROI recadrée sur la tumeur) :
    canal 0 = CT (fenêtré en HU) — texture des structures péri-tumorales (indices d'invasion)
    canal 1 = masque GTVp binaire — indique au réseau *où* est la tumeur primaire
La ROI est la boîte englobante du GTVp DILATÉE d'une marge (ROI_MARGIN_MM) : on garde
délibérément le voisinage de la tumeur, car c'est là que se lit l'invasion.

Cible binaire : T4 (positif) vs {T2, T3} (négatif). Les patients T0/T1 sont EXCLUS de
l'entraînement — `TN/T.py` sait déjà les distinguer par la taille.

Données / conventions reprises de `seg` : split matérialisé sur le disque
(`config.data_root/{train,test}/<PatientID>/`), CT `<id>__CT.nii.gz`, masque `<id>.nii.gz`
(GTVp=1, GTVn=2), labels dans le CSV clinique. On n'entraîne que sur le split `train` ; une
fraction stratifiée sert de validation interne. Sorties dans `results/` (poids) et `figures/`
(matrice de confusion), écrasées à chaque run — pas de versionnage (cf. politique du dépôt).

Lancement sur le cluster (depuis la racine du dépôt) :

    python -m TN.t4
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

import monai
from monai.data import Dataset, list_data_collate
from monai.networks.nets import resnet18
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, Lambdad,
    ScaleIntensityRanged, CropForegroundd, ResizeWithPadOrCropd, ConcatItemsd,
    RandFlipd, RandRotate90d, RandGaussianNoised, RandScaleIntensityd,
    RandShiftIntensityd, EnsureTyped,
)

from seg import config
from seg.split import case_ids, patient_dir

# === Cible & données ===
GTVP_LABEL = 1                                   # tumeur primaire dans le masque HECKTOR
POS_CLASS = "T4"                                 # classe positive (invasion)
NEG_CLASSES = ("T2", "T3")                       # classe négative
TRAIN_CLASSES = (*NEG_CLASSES, POS_CLASS)        # T0/T1 exclus (gérés par TN/T.py)

# === Prétraitement de la ROI ===
ROI_SPACING = (1.5, 1.5, 1.5)                    # rééchantillonnage isotrope (mm)
ROI_MARGIN_MM = 20.0                             # marge autour de la bbox tumorale (capte l'invasion)
ROI_SIZE = (96, 96, 96)                          # taille fixe du cube réseau (→ FOV 144 mm à 1.5 mm)
HU_MIN, HU_MAX = -200.0, 400.0                   # fenêtre CT : tissus mous + arêtes osseuses
IN_CHANNELS = 2                                  # CT + masque GTVp (mettre 1 pour CT seul)

# === Entraînement ===
BATCH_SIZE = 4
EPOCHS = 120
LR = 1e-4
WEIGHT_DECAY = 1e-5
VAL_FRACTION = 0.2                               # part du split train réservée à la validation interne
SEED = config.rf_seed                            # graine partagée du dépôt (42)

# === Sorties (écrasées à chaque run) ===
CKPT_PATH = os.path.join(config.results_dir, "t4_resnet18.pth")
FIG_PATH = "figures/confusion_T4.png"


def build_records(split: str) -> list:
    """(ct, mask, label 0/1, case_id) pour les patients `split` en classes T2/T3/T4 dont les
    volumes CT + masque existent. mask = chemin du label ; binarisé sur GTVp dans le pipeline."""
    truth = pd.read_csv(config.csv_path).set_index("PatientID")["T-stage"].to_dict()
    records, skipped = [], 0
    for case_id in case_ids(config, split):
        stage = truth.get(case_id)
        if stage not in TRAIN_CLASSES:            # None, T0 ou T1 → exclu
            continue
        pdir = patient_dir(config, split, case_id)
        ct = os.path.join(pdir, f"{case_id}__CT.nii.gz")
        mask = os.path.join(pdir, f"{case_id}.nii.gz")
        if not (os.path.exists(ct) and os.path.exists(mask)):
            skipped += 1
            continue
        records.append({"ct": ct, "mask": mask,
                        "label": int(stage == POS_CLASS), "case_id": case_id})
    n_pos = sum(r["label"] for r in records)
    print(f"[data] {split}: {len(records)} patients (T4={n_pos}, T2/T3={len(records) - n_pos}), "
          f"{skipped} ignorés (volume manquant)")
    return records


def build_transforms(train: bool) -> Compose:
    """Pipeline MONAI : chargement, orientation RAS, rééchantillonnage iso, fenêtrage CT,
    binarisation du masque GTVp, recadrage sur la ROI (bbox + marge), taille fixe, puis
    (train) augmentations légères, enfin concaténation CT⊕masque en un tenseur `image`."""
    margin_vox = int(round(ROI_MARGIN_MM / ROI_SPACING[0]))
    keys = ["ct", "mask"]
    xf = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=ROI_SPACING, mode=("bilinear", "nearest")),
        Lambdad(keys="mask", func=lambda x: (x == GTVP_LABEL).astype(np.float32)),  # GTVp binaire
        ScaleIntensityRanged(keys="ct", a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        # ROI = boîte englobante de la tumeur + marge (le voisinage porte le signe d'invasion)
        CropForegroundd(keys=keys, source_key="mask", margin=margin_vox, allow_smaller=True),
        ResizeWithPadOrCropd(keys=keys, spatial_size=ROI_SIZE),
    ]
    if train:
        xf += [
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=keys, prob=0.5, max_k=3),
            RandGaussianNoised(keys="ct", prob=0.2, std=0.02),
            RandScaleIntensityd(keys="ct", factors=0.1, prob=0.3),
            RandShiftIntensityd(keys="ct", offsets=0.1, prob=0.3),
        ]
    # canal 0 = CT, canal 1 = masque (ou CT seul si IN_CHANNELS == 1)
    concat = keys[:IN_CHANNELS]
    xf += [ConcatItemsd(keys=concat, name="image", dim=0), EnsureTyped(keys=["image"])]
    return Compose(xf)


def make_loaders(train_recs, val_recs):
    """DataLoaders train/val. Le train utilise un WeightedRandomSampler (inverse de la fréquence
    de classe) pour compenser le déséquilibre T4 vs T2/T3."""
    labels = np.array([r["label"] for r in train_recs])
    class_count = np.bincount(labels, minlength=2)
    sample_w = (1.0 / class_count)[labels]        # poids ∝ 1/effectif de la classe du patient
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                    num_samples=len(train_recs), replacement=True)
    train_loader = DataLoader(
        Dataset(train_recs, build_transforms(train=True)),
        batch_size=BATCH_SIZE, sampler=sampler, num_workers=config.num_workers,
        collate_fn=list_data_collate, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        Dataset(val_recs, build_transforms(train=False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=config.num_workers,
        collate_fn=list_data_collate, pin_memory=True)
    return train_loader, val_loader, class_count


@torch.no_grad()
def evaluate(model, loader, device):
    """Renvoie (probas T4, cibles) concaténées sur tout le loader."""
    model.eval()
    probs, targets = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        targets.append(batch["label"])
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def plot_confusion(y_true, y_pred, path, auc, bacc):
    """Matrice de confusion binaire {T2/T3, T4} annotée."""
    labels, names = [0, 1], ["T2/T3", "T4"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, names)
    ax.set_yticks(labels, names)
    ax.set_xlabel("Prédiction (ResNet-18 3D)")
    ax.set_ylabel("Vérité terrain")
    ax.set_title(f"Invasion T4 — AUC {auc:.3f} · bal.acc {bacc:.3f}")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in labels:
        for j in labels:
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"matrice de confusion → {path}")


def main():
    monai.utils.set_determinism(seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split train interne stratifié (le split disque `test` reste pour l'évaluation finale).
    records = build_records("train")
    strat = [r["label"] for r in records]
    train_recs, val_recs = train_test_split(
        records, test_size=VAL_FRACTION, random_state=SEED, stratify=strat)
    train_loader, val_loader, class_count = make_loaders(train_recs, val_recs)

    model = resnet18(spatial_dims=3, n_input_channels=IN_CHANNELS, num_classes=2).to(device)
    # Perte pondérée (inverse de la fréquence) — double garde-fou avec le sampler.
    class_w = torch.as_tensor(class_count.sum() / (2.0 * class_count), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_auc = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()

        probs, targets = evaluate(model, val_loader, device)
        auc = roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else float("nan")
        bacc = balanced_accuracy_score(targets, probs >= 0.5)
        print(f"epoch {epoch:3d}/{EPOCHS} · train loss {running / len(train_recs):.4f} · "
              f"val AUC {auc:.4f} · val bal.acc {bacc:.4f}")

        if auc > best_auc:                        # sélection du meilleur modèle sur l'AUC de validation
            best_auc = auc
            os.makedirs(config.results_dir, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch, "val_auc": auc, "val_balanced_acc": bacc,
                "in_channels": IN_CHANNELS, "roi_spacing": ROI_SPACING, "roi_size": ROI_SIZE,
                "roi_margin_mm": ROI_MARGIN_MM, "hu_window": (HU_MIN, HU_MAX),
                "pos_class": POS_CLASS, "neg_classes": NEG_CLASSES,
            }, CKPT_PATH)
            print(f"  ↳ meilleur AUC {best_auc:.4f} → {CKPT_PATH}")

    # Rapport final sur la validation avec le meilleur modèle.
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    probs, targets = evaluate(model, val_loader, device)
    auc = roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else float("nan")
    bacc = balanced_accuracy_score(targets, probs >= 0.5)
    print(f"\n[final] meilleur modèle (epoch {ckpt['epoch']}) · val AUC {auc:.4f} · "
          f"val bal.acc {bacc:.4f}")
    plot_confusion(targets, (probs >= 0.5).astype(int), FIG_PATH, auc, bacc)


if __name__ == "__main__":
    main()
