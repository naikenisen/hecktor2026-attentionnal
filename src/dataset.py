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
    "M-stage",
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
        # Dictionnaire de mapping valeur → entier pour chaque variable catégorielle
        self.cat_maps: Dict[str, Dict[str, int]] = {}

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
            # Valeurs uniques de la colonne avec ajout forcé de la classe inconnue
            vals = df[col].astype(str).fillna("Inconnu").unique().tolist()
            if "Inconnu" not in vals:
                vals.append("Inconnu")
            # Mapping alphabétique valeur → index entier
            self.cat_maps[col] = {v: i for i, v in enumerate(sorted(vals))}
        return self

    # Transforme une ligne du DataFrame en vecteur numpy (7,) de features normalisées
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
            # Valeur brute de la variable catégorielle (NaN → "Inconnu")
            val = row.get(col, np.nan)
            if pd.isna(val):
                val = "Inconnu"
            val = str(val)
            # Mapping de la colonne courante
            mapping = self.cat_maps[col]
            # Index entier de la catégorie (inconnue si absente du mapping)
            idx = mapping.get(val, mapping["Inconnu"])
            feats.append(float(idx))
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


# Dataset MONAI multitâche qui fournit images, cibles de segmentation et données tabulaires
class HECKTORMultitaskDataset(CacheDataset):

    # Délègue l'initialisation à CacheDataset avec les transforms multitâche
    def __init__(self, data_list, transform, cache_rate, num_workers):
        super().__init__(data=data_list, transform=transform,
                         cache_rate=cache_rate, num_workers=num_workers)

    # Retourne un sample enrichi avec les champs tabulaires (clinical, t_label, etc.)
    def __getitem__(self, idx):
        # Sample(s) fourni par CacheDataset après application des transforms
        item = super().__getitem__(idx)
        if isinstance(item, list):
            for s in item:
                s.update(self._tabular(s))
            return item
        item.update(self._tabular(item))
        return item

    # Stub de merge tabulaire : les champs sont déjà présents via le data_list initial
    @staticmethod
    def _tabular(sample):
        return {}


# Construit la liste de dictionnaires d'un split (train ou val) avec toutes les cibles
def _build_data_list(case_ids, images_dir, labels_dir, df, clinical_encoder) -> List[dict]:
    # Liste des dicts à retourner, un par patient
    items = []
    # Indexation du DataFrame par PatientID pour un accès O(1)
    df_idx = df.set_index("PatientID")
    for cid in case_ids:
        if cid not in df_idx.index:
            continue
        # Ligne CSV du patient courant
        row = df_idx.loc[cid]
        # Vecteur clinique encodé (7,)
        clin = clinical_encoder.transform_row(row)
        # Label T encodé en entier (−1 si inconnu)
        t_lbl = _encode_stage(row.get("T_stage"), T_STAGES)
        # Label N encodé en entier (−1 si inconnu)
        n_lbl = _encode_stage(row.get("N_stage"), N_STAGES)
        # Temps de suivi RFS en valeur flottante (0.0 si absent)
        rfs = float(row.get("RFS", np.nan)) if not pd.isna(row.get("RFS", np.nan)) else 0.0
        # Indicateur d'événement (1 = rechute, 0 = censuré)
        evt = int(row.get("Relapse", 0)) if not pd.isna(row.get("Relapse", np.nan)) else 0
        items.append({
            # Chemin vers le fichier CT .npz
            "ct":       os.path.join(images_dir, f"{cid}_ct.npz"),
            # Chemin vers le fichier PET .npz
            "pet":      os.path.join(images_dir, f"{cid}_pet.npz"),
            # Chemin vers le masque de segmentation .npz
            "label":    os.path.join(labels_dir, f"{cid}_label.npz"),
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


# Crée les DataLoaders train et val ainsi que le DataFrame d'entraînement pour les quantiles
def get_multitask_dataloaders(config) -> tuple:
    import random

    # Chemin complet vers le dossier des images CT/PET preprocessées
    images_dir = os.path.join(config.data_root, config.train_images_dir)
    # Chemin complet vers le dossier des labels de segmentation
    labels_dir = os.path.join(config.data_root, config.train_labels_dir)
    # Liste triée des fichiers CT pour déduire les case_ids disponibles
    ct_files = sorted(f for f in os.listdir(images_dir) if f.endswith("_ct.npz"))
    # Identifiants de cas extraits des noms de fichiers CT
    case_ids = [f.replace("_ct.npz", "") for f in ct_files]

    random.seed(config.seed)
    random.shuffle(case_ids)
    # Nombre de cas réservés à la validation
    n_val = int(len(case_ids) * config.val_split)
    # Identifiants du split de validation
    val_ids = case_ids[:n_val]
    # Identifiants du split d'entraînement
    train_ids = case_ids[n_val:]
    print(f"[Data] {len(train_ids)} train / {len(val_ids)} val")

    # DataFrame complet des données cliniques et cibles
    df = pd.read_csv(config.csv_path)

    # Encodeur clinique fitté uniquement sur les données d'entraînement
    clin_enc = ClinicalEncoder().fit(df[df["PatientID"].isin(train_ids)])

    # Liste de dicts des patients d'entraînement
    train_items = _build_data_list(train_ids, images_dir, labels_dir, df, clin_enc)
    # Liste de dicts des patients de validation
    val_items = _build_data_list(val_ids, images_dir, labels_dir, df, clin_enc)

    # Sous-DataFrame d'entraînement utilisé pour calculer les quantiles de temps
    train_df = df[df["PatientID"].isin([it["case_id"] for it in train_items])].copy()

    # Dataset MONAI mis en cache pour l'entraînement
    train_ds = HECKTORMultitaskDataset(
        data_list=train_items,
        transform=get_train_transforms(config),
        cache_rate=config.cache_rate,
        num_workers=config.num_workers,
    )
    # Dataset MONAI mis en cache pour la validation
    val_ds = HECKTORMultitaskDataset(
        data_list=val_items,
        transform=get_validation_transforms(),
        cache_rate=config.cache_rate,
        num_workers=config.num_workers,
    )

    # DataLoader d'entraînement avec shuffle et drop_last pour les batchs incomplets
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    # DataLoader de validation sans shuffle
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader, train_df, clin_enc
