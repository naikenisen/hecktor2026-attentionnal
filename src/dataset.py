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
def _encode_stage(value, stages: List[str]) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -1
    # Normalise en majuscules et supprime les espaces
    s = str(value).strip().upper()
    # Collapse les sous-types N2a/N2b/N2c en N2
    if s.startswith("N2"):
        s = "N2"
    return stages.index(s) if s in stages else -1


# Construit la liste de dicts d'un split : uniquement chemins d'images + identifiant.
# Les cibles cliniques ne dépendent pas du backbone et sont assemblées séparément
# (voir build_clinical_targets), donc elles ne transitent plus par le dataloader image.
def _build_data_list(case_ids, data_root, df) -> List[dict]:
    # Patients présents dans le CSV (les autres sont ignorés)
    known = set(df["PatientID"])
    items = []
    for cid in case_ids:
        if cid not in known:
            continue
        # Dossier du patient : {data_root}/{cid}/
        patient_dir = os.path.join(data_root, cid)
        items.append({
            # Chemin vers le fichier CT : {cid}/{cid}__CT.nii.gz
            "ct":      os.path.join(patient_dir, f"{cid}__CT.nii.gz"),
            # Chemin vers le fichier PET : {cid}/{cid}__PT.nii.gz
            "pet":     os.path.join(patient_dir, f"{cid}__PT.nii.gz"),
            # Chemin vers le masque de segmentation : {cid}/{cid}.nii.gz
            "label":   os.path.join(patient_dir, f"{cid}.nii.gz"),
            # Identifiant patient (clé de jointure avec les cibles cliniques)
            "case_id": cid,
        })
    return items


# Encode les cibles tabulaires d'un patient (clinique + T/N + survie) en tensors
def _encode_targets(row, clin_enc) -> dict:
    # Temps de suivi RFS (0.0 si absent)
    rfs = float(row.get("RFS", np.nan)) if not pd.isna(row.get("RFS", np.nan)) else 0.0
    # Indicateur d'événement (1 = rechute, 0 = censuré)
    evt = int(row.get("Relapse", 0)) if not pd.isna(row.get("Relapse", np.nan)) else 0
    return {
        # Vecteur clinique encodé (output_dim,)
        "clinical": torch.from_numpy(clin_enc.transform_row(row)),
        # Classe de staging T (−1 si inconnu)
        "t_label":  torch.tensor(_encode_stage(row.get("T-stage"), T_STAGES), dtype=torch.long),
        # Classe de staging N (−1 si inconnu)
        "n_label":  torch.tensor(_encode_stage(row.get("N-stage"), N_STAGES), dtype=torch.long),
        "time":     torch.tensor(rfs, dtype=torch.float32),
        "event":    torch.tensor(evt, dtype=torch.float32),
    }


# Assemble les tensors cliniques/cibles d'une liste de patients, empilés dans l'ordre
# des case_ids fournis (typiquement l'ordre des bottlenecks extraits). 100 % CPU/CSV :
# aucune dépendance au backbone. C'est le pendant clinique de l'extraction de features.
def build_clinical_targets(case_ids, df, clin_enc) -> dict:
    # Indexation du DataFrame par PatientID pour un accès O(1)
    df_idx = df.set_index("PatientID")
    # Une ligne encodée par patient, dans l'ordre fourni
    rows = [_encode_targets(df_idx.loc[cid], clin_enc) for cid in case_ids]
    # Empile chaque champ tensoriel sur la dimension batch
    out = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
    # Conserve l'ordre des identifiants pour la traçabilité / vérification d'alignement
    out["case_id"] = list(case_ids)
    return out


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

    train_items = _build_data_list(train_ids, config.data_root, df)
    val_items = _build_data_list(val_ids, config.data_root, df)

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
# Ne porte que les images + case_id ; les cibles cliniques sont assemblées en phase 2.
def get_feature_extraction_loaders(config) -> tuple:
    # Mêmes splits que l'entraînement de segmentation
    train_ids, val_ids = split_case_ids(config)
    print(f"[Extract] {len(train_ids)} train / {len(val_ids)} val")

    df = pd.read_csv(config.csv_path)
    train_items = _build_data_list(train_ids, config.data_root, df)
    val_items = _build_data_list(val_ids, config.data_root, df)

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
    return train_loader, val_loader
