"""Extraction des features CT-FM (project-lighter/ct_fm_feature_extractor).

Encode chaque CT par un SegResNet pré-entraîné en self-supervised contrastif sur 148 000
scanners (Imaging Data Commons) et en tire un vecteur figé de 512 caractéristiques (global
average pooling du dernier feature map de l'encodeur). L'extraction est idempotente : elle
n'utilise le GPU qu'à la première exécution, puis met les features en cache sur disque.
"""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config


def _build_preprocess():
    """Pipeline de prétraitement officiel de CT-FM, appliqué à un chemin de CT brute :
    orientation standardisée, fenêtrage HU [-1024, 2048] → [0, 1], puis recadrage sur le
    premier plan pour réduire le calcul. (Imports MONAI locaux : non requis quand les
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
            ct_path = _ct_path(self.config, case_id)
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


def _ct_path(config, case_id: str) -> str:
    return os.path.join(config.data_root, case_id, f"{case_id}__CT.nii.gz")


def _expected_ids(config, case_ids: list) -> list:
    """case_id du split dont la CT existe : exactement ce que `embed_split` produira."""
    return [c for c in case_ids if os.path.exists(_ct_path(config, c))]


def ensure_ct_fm_features(config):
    """Extrait les embeddings CT-FM des deux splits si le cache ne couvre pas déjà le split
    courant. Idempotent tant que `data_root` est stable ; sinon (cache périmé) il réextrait
    automatiquement. N'utilise le GPU que lors d'une (ré)extraction effective."""
    from src.split import split_case_ids  # source unique du split (CPU seul, sans MONAI)
    from src.feature_cache import cache_covers
    train_ids, val_ids = split_case_ids(config)
    exp_train, exp_val = _expected_ids(config, train_ids), _expected_ids(config, val_ids)
    if (cache_covers(config.foundation_train_features_path, exp_train)
            and cache_covers(config.foundation_val_features_path, exp_val)):
        print(f"CT-FM features already extracted for current split "
              f"({len(exp_train)} train / {len(exp_val)} val), skipping")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = CtFmExtractor(config, device)
    os.makedirs(config.features_dir, exist_ok=True)
    print(f"extracting CT-FM train split ({len(exp_train)} cases)")
    torch.save(extractor.embed_split(train_ids), config.foundation_train_features_path)
    print(f"extracting CT-FM val split ({len(exp_val)} cases)")
    torch.save(extractor.embed_split(val_ids), config.foundation_val_features_path)
    print(f"CT-FM features saved to {config.features_dir}")
