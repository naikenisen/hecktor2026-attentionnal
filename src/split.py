"""Split train/val déterministe des patients — source unique partagée par toute la pipeline.

Utilisé par la segmentation (`seg`), l'extraction d'embeddings T/N (`tn`), la survie clinique
(`survival`) et la variante CT-FM (`foundation_survival`) : garantit des splits strictement
identiques entre toutes les têtes.
"""
import os
import random


def split_ids(case_ids: list, config) -> tuple[list, list]:
    """Split train/val déterministe sur une liste de case_ids fournie. Renvoie (train_ids, val_ids)."""
    case_ids = sorted(case_ids)
    random.seed(config.seed)
    random.shuffle(case_ids)
    n_val = int(len(case_ids) * config.val_split)
    return case_ids[n_val:], case_ids[:n_val]


def split_case_ids(config) -> tuple[list, list]:
    """Split train/val déterministe des patients présents dans `data_root` (CPU seul, sans
    MONAI). Renvoie (train_ids, val_ids)."""
    case_ids = [
        d for d in os.listdir(config.data_root)
        if os.path.isdir(os.path.join(config.data_root, d))
    ]
    return split_ids(case_ids, config)
