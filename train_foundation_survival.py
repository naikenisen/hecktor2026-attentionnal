"""Prédiction de la survie sans rechute par RandomSurvivalForest sur l'embedding CT
extrait par le modèle de fondation CT-FM (project-lighter/ct_fm_feature_extractor).

Contrairement à train_survival.py (variables cliniques seules) et train_tn.py
(embedding TEP/CT du backbone de segmentation), ce script n'utilise QUE la CT, encodée
par un SegResNet pré-entraîné en self-supervised contrastif sur 148 000 scanners de
l'Imaging Data Commons. Pour chaque patient on extrait un vecteur figé de 512
caractéristiques (global average pooling du dernier feature map de l'encodeur), sur
lequel on entraîne un RandomSurvivalForest. Cible : la paire (événement, RFS).

Le split train/val réutilise exactement celui de la pipeline image (split_case_ids,
même seed) pour rester comparable à train_survival.py. L'extraction des embeddings est
idempotente : elle n'utilise le GPU qu'à la première exécution, puis met les features
en cache sur disque. Recherche Optuna des hyperparamètres sur le c-index de validation."""
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import optuna
from tqdm import tqdm
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

import config
from utils.metrics import c_index


# ───────────────────────────── Extraction des features CT-FM ─────────────────────────────

def _build_preprocess():
    """Pipeline de prétraitement officiel de CT-FM, appliqué à un chemin de CT brute :
    orientation standardisée, fenêtrage HU [-1024, 2048] → [0, 1], puis recadrage sur
    le premier plan pour réduire le calcul. (Imports MONAI locaux : non requis quand les
    features sont déjà en cache.)"""
    from monai.transforms import (
        Compose, LoadImage, EnsureType, Orientation, ScaleIntensityRange, CropForeground,
    )
    return Compose([
        LoadImage(ensure_channel_first=True),   # charge la NIfTI et garantit la dim de canal
        EnsureType(),
        Orientation(axcodes="SPL"),             # orientation attendue par CT-FM
        ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0.0, b_max=1.0, clip=True),
        CropForeground(allow_smaller=True),     # retire le fond pour alléger le calcul
    ])


class CtFmExtractor:
    """Charge l'encodeur SegResNet pré-entraîné de CT-FM et produit, pour chaque patient,
    le vecteur de 512 caractéristiques figé de sa CT (global average pooling)."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.preprocess = _build_preprocess()
        from lighter_zoo import SegResEncoder
        self.model = SegResEncoder.from_pretrained(config.foundation_model_id).to(device).eval()

    @torch.no_grad()
    def _embed(self, ct_path: str, device) -> torch.Tensor:
        x = self.preprocess(ct_path).unsqueeze(0).to(device)         # (1, 1, D, H, W)
        feature_map = self.model.to(device)(x)[-1]                   # dernier feature map (1, 512, d, h, w)
        # Average pooling : compresse le feature map en un vecteur de 512 par patient.
        return F.adaptive_avg_pool3d(feature_map, 1).flatten().float().cpu()

    @torch.no_grad()
    def embed_split(self, case_ids: list) -> dict:
        embeddings, kept = [], []
        for case_id in tqdm(case_ids, desc="CT-FM", leave=False):
            ct_path = os.path.join(self.config.data_root, case_id, f"{case_id}__CT.nii.gz")
            if not os.path.exists(ct_path):
                continue
            try:
                embeddings.append(self._embed(ct_path, self.device))
            except RuntimeError as err:
                if "out of memory" not in str(err).lower():
                    raise
                # Volume trop volumineux pour le GPU : repli CPU pour ce patient.
                torch.cuda.empty_cache()
                embeddings.append(self._embed(ct_path, torch.device("cpu")))
                self.model.to(self.device)
            kept.append(case_id)
        return {"embedding": torch.stack(embeddings), "case_id": kept}


def ensure_ct_fm_features(config):
    """Extrait les embeddings CT-FM des deux splits si les fichiers de features sont
    absents. Idempotent : n'utilise le GPU que lors de la première extraction."""
    if (os.path.exists(config.foundation_train_features_path)
            and os.path.exists(config.foundation_val_features_path)):
        print("CT-FM features already extracted, skipping")
        return
    from src.clinical_data import split_case_ids  # source unique du split (CPU seul, sans MONAI)
    train_ids, val_ids = split_case_ids(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = CtFmExtractor(config, device)
    os.makedirs(config.features_dir, exist_ok=True)
    print("extracting CT-FM train split")
    torch.save(extractor.embed_split(train_ids), config.foundation_train_features_path)
    print("extracting CT-FM val split")
    torch.save(extractor.embed_split(val_ids), config.foundation_val_features_path)
    print(f"CT-FM features saved to {config.features_dir}")


# ──────────────────────────── Jeu de survie aligné sur l'embedding ────────────────────────────

def _survival_xy(features: dict, patients: pd.DataFrame) -> tuple:
    """Aligne l'embedding CT-FM de chaque case_id avec sa cible de survie (temps, événement).
    Ne conserve que les patients dont le RFS est renseigné (> 0)."""
    rows = patients.set_index("PatientID")
    embeddings = np.asarray(features["embedding"], dtype=np.float32)
    X, time, event = [], [], []
    for i, case_id in enumerate(features["case_id"]):
        if case_id not in rows.index:
            continue
        row = rows.loc[case_id]
        relapse_free_survival = row.get("RFS", np.nan)
        if pd.isna(relapse_free_survival) or float(relapse_free_survival) <= 0:
            continue
        X.append(embeddings[i])
        time.append(float(relapse_free_survival))
        relapse = row.get("Relapse", np.nan)
        event.append(int(relapse) if not pd.isna(relapse) else 0)
    return np.stack(X), np.asarray(time, dtype=np.float64), np.asarray(event, dtype=bool)


def load_foundation_survival(config) -> tuple:
    """Charge les embeddings CT-FM figés des deux splits et les joint à la cible de survie."""
    # weights_only=False : features produites par notre propre CtFmExtractor.
    train = torch.load(config.foundation_train_features_path, map_location="cpu", weights_only=False)
    val = torch.load(config.foundation_val_features_path, map_location="cpu", weights_only=False)
    patients = pd.read_csv(config.csv_path)
    train_xy = _survival_xy(train, patients)
    val_xy = _survival_xy(val, patients)
    print(f"CT-FM survival: {len(train_xy[1])} train / {len(val_xy[1])} val patients with RFS "
          f"(dim {train_xy[0].shape[1]})")
    return train_xy, val_xy


# ─────────────────────────────── RandomSurvivalForest ───────────────────────────────

def _build_rsf(params: dict) -> RandomSurvivalForest:
    return RandomSurvivalForest(
        random_state=config.rf_seed,
        n_jobs=-1,
        **params,
    )


def main():
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ensure_ct_fm_features(config)
    (X_train, t_train, e_train), (X_val, t_val, e_val) = load_foundation_survival(config)
    y_train = Surv.from_arrays(event=e_train, time=t_train)

    def objective(trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 15),
        )
        rsf = _build_rsf(params)
        rsf.fit(X_train, y_train)
        # RSF.predict renvoie un score de risque (croissant avec le risque) :
        # c_index attend un score de risque et applique lui-même la négation.
        return c_index(rsf.predict(X_val), t_val, e_val.astype(float))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.rf_seed),
    )
    study.optimize(objective, n_trials=config.surv_n_trials)

    best = _build_rsf(study.best_params)
    best.fit(X_train, y_train)
    joblib.dump(best, config.best_foundation_survival_path)
    print(f"\nbest c-index {study.best_value:.4f}")
    print(f"best params {study.best_params}")
    print(f"model saved to {config.best_foundation_survival_path}")


if __name__ == "__main__":
    main()
