import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict

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


# Charge un split de features pré-calculées (bottleneck + case_id) depuis un .pt
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
