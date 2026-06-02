import os
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict
from monai.data import CacheDataset, DataLoader

from src.transforms import (
    get_multitask_train_transforms as get_train_transforms,
    get_multitask_validation_transforms as get_validation_transforms,
)

# Colonne continue utilisée comme feature clinique (Age)
CLINICAL_NUMERIC = ["Age"]
# Colonnes catégorielles utilisées comme features cliniques (6 variables)
CLINICAL_CATEGORICAL = [
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "HPV Status",
    "Treatment",
]
# Vérifie que le total des features cliniques vaut bien 7 comme attendu par le MLP
assert len(CLINICAL_NUMERIC) + len(CLINICAL_CATEGORICAL) == 7

# Labels du staging T dans l'ordre d'encodage ordinal
T_STAGES = ["T1", "T2", "T3", "T4"]
# Labels du staging N dans l'ordre d'encodage ordinal
N_STAGES = ["N0", "N1", "N2", "N3"]


# Encode les 7 colonnes cliniques en un vecteur numérique standardisé
class ClinicalEncoder:

    # Initialise les statistiques et mappings à None avant le fit
    def __init__(self):
        # Médiane de l'âge utilisée pour l'imputation des NaN
        self.age_median: Optional[float] = None
        # Moyenne de l'âge pour la standardisation
        self.age_mean: Optional[float] = None
        # Écart-type de l'âge pour la standardisation
        self.age_std: Optional[float] = None
        # Dictionnaire de mapping valeur → index one-hot pour chaque variable catégorielle
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        # Dimension totale du vecteur encodé (1 âge + Σ one-hot) ; fixée au fit
        self.output_dim: Optional[int] = None

    # Calcule les statistiques d'encodage sur le DataFrame d'entraînement
    def fit(self, df: pd.DataFrame):
        # Valeurs d'âge sans NaN pour calculer les statistiques
        ages = df["Age"].dropna().values.astype(float)
        # Médiane de l'âge pour l'imputation
        self.age_median = float(np.median(ages)) if len(ages) else 0.0
        # Moyenne pour centrer l'âge
        self.age_mean = float(ages.mean()) if len(ages) else 0.0
        # Écart-type pour normaliser l'âge (protection contre division par zéro)
        self.age_std = float(ages.std()) if len(ages) and ages.std() > 1e-6 else 1.0

        for col in CLINICAL_CATEGORICAL:
            # Catégories observées (NaN exclu) + classe "Inconnu" pour les manquants
            vals = df[col].dropna().astype(str).unique().tolist()
            if "Inconnu" not in vals:
                vals.append("Inconnu")
            # Mapping ordonné valeur → position dans le vecteur one-hot
            self.cat_maps[col] = {v: i for i, v in enumerate(sorted(vals))}

        # Dimension finale : âge (1) + somme des cardinalités one-hot
        self.output_dim = 1 + sum(len(m) for m in self.cat_maps.values())
        return self

    # Transforme une ligne du DataFrame en vecteur numpy (output_dim,) normalisé/one-hot
    def transform_row(self, row: pd.Series) -> np.ndarray:
        # Valeur brute de l'âge (NaN remplacé par la médiane)
        age = row.get("Age", np.nan)
        if pd.isna(age):
            age = self.age_median
        # Âge centré et réduit (z-score)
        age_z = (float(age) - self.age_mean) / self.age_std

        # Initialise le vecteur de features avec l'âge normalisé
        feats = [age_z]
        for col in CLINICAL_CATEGORICAL:
            # Mapping de la colonne courante
            mapping = self.cat_maps[col]
            # Valeur brute de la variable catégorielle (NaN → "Inconnu")
            val = row.get(col, np.nan)
            val = "Inconnu" if pd.isna(val) else str(val)
            # Index one-hot (catégorie inconnue au train → "Inconnu")
            idx = mapping.get(val, mapping["Inconnu"])
            # Bloc one-hot de la variable
            onehot = [0.0] * len(mapping)
            onehot[idx] = 1.0
            feats.extend(onehot)
        return np.array(feats, dtype=np.float32)


# Convertit un label T ou N brut en entier (retourne -1 si inconnu pour ignore_index PyTorch)
def _encode_stage(value, stages: List[str], unknown_idx: int = 0) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -1
    # Normalise en majuscules et supprime les espaces
    s = str(value).strip().upper()
    # Collapse les sous-types N2a/N2b/N2c en N2
    if s.startswith("N2"):
        s = "N2"
    return stages.index(s) if s in stages else -1


# Construit la liste de dictionnaires d'un split (train ou val) avec toutes les cibles
def _build_data_list(case_ids, data_root, df, clinical_encoder) -> List[dict]:
    # Liste des dicts à retourner, un par patient
    items = []
    # Indexation du DataFrame par PatientID pour un accès O(1)
    df_idx = df.set_index("PatientID")
    for cid in case_ids:
        if cid not in df_idx.index:
            continue
        # Dossier du patient : {data_root}/{cid}/
        patient_dir = os.path.join(data_root, cid)
        # Ligne CSV du patient courant
        row = df_idx.loc[cid]
        # Vecteur clinique encodé (7,)
        clin = clinical_encoder.transform_row(row)
        # Label T encodé en entier (−1 si inconnu)
        t_lbl = _encode_stage(row.get("T-stage"), T_STAGES)
        # Label N encodé en entier (−1 si inconnu)
        n_lbl = _encode_stage(row.get("N-stage"), N_STAGES)
        # Temps de suivi RFS en valeur flottante (0.0 si absent)
        rfs = float(row.get("RFS", np.nan)) if not pd.isna(row.get("RFS", np.nan)) else 0.0
        # Indicateur d'événement (1 = rechute, 0 = censuré)
        evt = int(row.get("Relapse", 0)) if not pd.isna(row.get("Relapse", np.nan)) else 0
        items.append({
            # Chemin vers le fichier CT : {cid}/{cid}__CT.nii.gz
            "ct":       os.path.join(patient_dir, f"{cid}__CT.nii.gz"),
            # Chemin vers le fichier PET : {cid}/{cid}__PT.nii.gz
            "pet":      os.path.join(patient_dir, f"{cid}__PT.nii.gz"),
            # Chemin vers le masque de segmentation : {cid}/{cid}.nii.gz
            "label":    os.path.join(patient_dir, f"{cid}.nii.gz"),
            # Vecteur clinique encodé (7,) sous forme de tensor
            "clinical": torch.from_numpy(clin),
            # Classe de staging T comme tensor long
            "t_label":  torch.tensor(t_lbl, dtype=torch.long),
            # Classe de staging N comme tensor long
            "n_label":  torch.tensor(n_lbl, dtype=torch.long),
            # Temps de suivi RFS comme tensor float
            "time":     torch.tensor(rfs, dtype=torch.float32),
            # Indicateur d'événement comme tensor float
            "event":    torch.tensor(evt, dtype=torch.float32),
            # Identifiant patient pour le débogage et la traçabilité
            "case_id":  cid,
        })
    return items


# Découpe les patients en splits train/val de façon déterministe (même seed partout)
def split_case_ids(config) -> tuple:
    import random

    # Découverte des patients : chaque sous-dossier de data_root est un patient
    case_ids = sorted(
        d for d in os.listdir(config.data_root)
        if os.path.isdir(os.path.join(config.data_root, d))
    )

    random.seed(config.seed)
    random.shuffle(case_ids)
    # Nombre de cas réservés à la validation
    n_val = int(len(case_ids) * config.val_split)
    # (train_ids, val_ids)
    return case_ids[n_val:], case_ids[:n_val]


# Crée les DataLoaders train et val pour l'entraînement de la segmentation
def get_seg_dataloaders(config) -> tuple:
    train_ids, val_ids = split_case_ids(config)
    print(f"[Data] {len(train_ids)} train / {len(val_ids)} val")

    df = pd.read_csv(config.csv_path)

    clin_enc = ClinicalEncoder().fit(df[df["PatientID"].isin(train_ids)])
    config.n_clinical_features = clin_enc.output_dim
    print(f"[Data] dim features cliniques (one-hot) = {clin_enc.output_dim}")

    train_items = _build_data_list(train_ids, config.data_root, df, clin_enc)
    val_items = _build_data_list(val_ids, config.data_root, df, clin_enc)

    train_ds = CacheDataset(
        data=train_items,
        transform=get_train_transforms(config),
        cache_rate=config.cache_rate,
        num_workers=config.num_workers,
    )
    val_ds = CacheDataset(
        data=val_items,
        transform=get_validation_transforms(config),
        cache_rate=config.cache_rate,
        num_workers=config.num_workers,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader


# Crée des loaders déterministes (transforms de validation, sans shuffle) pour
# l'extraction de features : le backbone est figé, on veut un bottleneck reproductible.
def get_feature_extraction_loaders(config) -> tuple:
    # Mêmes splits que l'entraînement multitâche
    train_ids, val_ids = split_case_ids(config)
    print(f"[Extract] {len(train_ids)} train / {len(val_ids)} val")

    # DataFrame complet des cibles
    df = pd.read_csv(config.csv_path)
    # Encodeur clinique fitté uniquement sur le train (identique à l'entraînement)
    clin_enc = ClinicalEncoder().fit(df[df["PatientID"].isin(train_ids)])
    # Propage la dimension one-hot réelle à la config
    config.n_clinical_features = clin_enc.output_dim
    print(f"[Extract] dim features cliniques (one-hot) = {clin_enc.output_dim}")

    # Listes de dicts par split
    train_items = _build_data_list(train_ids, config.data_root, df, clin_enc)
    val_items = _build_data_list(val_ids, config.data_root, df, clin_enc)
    # Sous-DataFrame train pour les quantiles de bins temporels
    train_df = df[df["PatientID"].isin([it["case_id"] for it in train_items])].copy()

    tf = get_validation_transforms(config)
    train_ds = CacheDataset(data=train_items, transform=tf,
                            cache_rate=config.cache_rate, num_workers=config.num_workers)
    val_ds = CacheDataset(data=val_items, transform=tf,
                          cache_rate=config.cache_rate, num_workers=config.num_workers)

    # Loaders sans shuffle ni drop_last : on veut tous les patients, dans l'ordre
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_df, clin_enc
