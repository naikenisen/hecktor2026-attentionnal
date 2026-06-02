import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict

CLINICAL_NUMERIC = ["Age"]

CLINICAL_CATEGORICAL = [
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "HPV Status",
    "Treatment",
]

assert len(CLINICAL_NUMERIC) + len(CLINICAL_CATEGORICAL) == 7

T_STAGES = ["T1", "T2", "T3", "T4"]

N_STAGES = ["N0", "N1", "N2", "N3"]

class ClinicalEncoder:

    def __init__(self):

        self.age_median: Optional[float] = None

        self.age_mean: Optional[float] = None

        self.age_std: Optional[float] = None

        self.cat_maps: Dict[str, Dict[str, int]] = {}

        self.output_dim: Optional[int] = None

    def fit(self, df: pd.DataFrame):

        ages = df["Age"].dropna().values.astype(float)

        self.age_median = float(np.median(ages)) if len(ages) else 0.0

        self.age_mean = float(ages.mean()) if len(ages) else 0.0

        self.age_std = float(ages.std()) if len(ages) and ages.std() > 1e-6 else 1.0

        for col in CLINICAL_CATEGORICAL:

            vals = df[col].dropna().astype(str).unique().tolist()
            if "Inconnu" not in vals:
                vals.append("Inconnu")

            self.cat_maps[col] = {v: i for i, v in enumerate(sorted(vals))}

        self.output_dim = 1 + sum(len(m) for m in self.cat_maps.values())
        return self

    def transform_row(self, row: pd.Series) -> np.ndarray:

        age = row.get("Age", np.nan)
        if pd.isna(age):
            age = self.age_median

        age_z = (float(age) - self.age_mean) / self.age_std

        feats = [age_z]
        for col in CLINICAL_CATEGORICAL:

            mapping = self.cat_maps[col]

            val = row.get(col, np.nan)
            val = "Inconnu" if pd.isna(val) else str(val)

            idx = mapping.get(val, mapping["Inconnu"])

            onehot = [0.0] * len(mapping)
            onehot[idx] = 1.0
            feats.extend(onehot)
        return np.array(feats, dtype=np.float32)

def _encode_stage(value, stages: List[str]) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -1

    s = str(value).strip().upper()

    if s.startswith("N2"):
        s = "N2"
    return stages.index(s) if s in stages else -1

def _encode_targets(row, clin_enc) -> dict:

    rfs = float(row.get("RFS", np.nan)) if not pd.isna(row.get("RFS", np.nan)) else 0.0

    evt = int(row.get("Relapse", 0)) if not pd.isna(row.get("Relapse", np.nan)) else 0
    return {

        "clinical": torch.from_numpy(clin_enc.transform_row(row)),

        "t_label":  torch.tensor(_encode_stage(row.get("T-stage"), T_STAGES), dtype=torch.long),

        "n_label":  torch.tensor(_encode_stage(row.get("N-stage"), N_STAGES), dtype=torch.long),
        "time":     torch.tensor(rfs, dtype=torch.float32),
        "event":    torch.tensor(evt, dtype=torch.float32),
    }

def build_clinical_targets(case_ids, df, clin_enc) -> dict:

    df_idx = df.set_index("PatientID")

    rows = [_encode_targets(df_idx.loc[cid], clin_enc) for cid in case_ids]

    out = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}

    out["case_id"] = list(case_ids)
    return out

def compute_bin_edges(train_df, n_bins: int) -> np.ndarray:

    ev = train_df[train_df["Relapse"] == 1]["RFS"].dropna().values.astype(float)
    if len(ev) < n_bins:

        ev = train_df["RFS"].dropna().values.astype(float)
    if len(ev) == 0:
        return np.linspace(0, 1, n_bins + 1)

    edges = np.quantile(ev, np.linspace(0, 1, n_bins + 1))

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def load_features(path: str, device) -> dict:
    d = torch.load(path, map_location="cpu")

    return d

def class_weights(labels: torch.Tensor, num_classes: int, device) -> torch.Tensor:
    valid = labels[labels >= 0]
    counts = torch.bincount(valid, minlength=num_classes).float().clamp_min(1.0)

    w = counts.sum() / (num_classes * counts)
    return w.to(device)
