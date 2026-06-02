import numpy as np
import pandas as pd
import torch

AGE_COLUMN = "Age"
CATEGORICAL_COLUMNS = [
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "HPV Status",
    "Treatment",
]
T_STAGES = ["T1", "T2", "T3", "T4"]
N_STAGES = ["N0", "N1", "N2", "N3"]
UNKNOWN_CATEGORY = "Inconnu"
UNKNOWN_STAGE = -1

class ClinicalEncoder:

    def __init__(self):
        self.age_mean = 0.0
        self.age_std = 1.0
        self.age_median = 0.0
        self.category_indices = {}

    def fit(self, patients: pd.DataFrame):
        ages = patients[AGE_COLUMN].dropna().to_numpy(dtype=float)
        if len(ages):
            self.age_median = float(np.median(ages))
            self.age_mean = float(ages.mean())
            self.age_std = float(ages.std()) if ages.std() > 1e-6 else 1.0
        for column in CATEGORICAL_COLUMNS:
            values = patients[column].dropna().astype(str).unique().tolist()
            if UNKNOWN_CATEGORY not in values:
                values.append(UNKNOWN_CATEGORY)
            self.category_indices[column] = {value: i for i, value in enumerate(sorted(values))}
        return self

    @property
    def output_dim(self) -> int:
        return 1 + sum(len(indices) for indices in self.category_indices.values())

    def encode_row(self, row: pd.Series) -> np.ndarray:
        age = row.get(AGE_COLUMN, np.nan)
        age = self.age_median if pd.isna(age) else float(age)
        features = [(age - self.age_mean) / self.age_std]
        for column, indices in self.category_indices.items():
            value = row.get(column, np.nan)
            value = UNKNOWN_CATEGORY if pd.isna(value) else str(value)
            position = indices.get(value, indices[UNKNOWN_CATEGORY])
            one_hot = [0.0] * len(indices)
            one_hot[position] = 1.0
            features.extend(one_hot)
        return np.array(features, dtype=np.float32)

class ClinicalDataset:

    TENSOR_FIELDS = ("bottleneck", "clinical", "t_label", "n_label", "time", "event")

    def __init__(self, tensors: dict, case_ids: list):
        self.tensors = tensors
        self.case_ids = case_ids

    @staticmethod
    def _stage_index(value, stages: list) -> int:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return UNKNOWN_STAGE
        label = str(value).strip().upper()
        if label.startswith("N2"):
            label = "N2"
        return stages.index(label) if label in stages else UNKNOWN_STAGE

    @classmethod
    def from_bottlenecks(cls, features: dict, patients: pd.DataFrame, encoder: ClinicalEncoder):
        rows = patients.set_index("PatientID")
        clinical, t_label, n_label, time, event = [], [], [], [], []
        for case_id in features["case_id"]:
            row = rows.loc[case_id]
            clinical.append(torch.from_numpy(encoder.encode_row(row)))
            t_label.append(cls._stage_index(row.get("T-stage"), T_STAGES))
            n_label.append(cls._stage_index(row.get("N-stage"), N_STAGES))
            relapse_free_survival = row.get("RFS", np.nan)
            time.append(float(relapse_free_survival) if not pd.isna(relapse_free_survival) else 0.0)
            relapse = row.get("Relapse", np.nan)
            event.append(int(relapse) if not pd.isna(relapse) else 0)
        tensors = {
            "bottleneck": features["bottleneck"],
            "clinical": torch.stack(clinical),
            "t_label": torch.tensor(t_label, dtype=torch.long),
            "n_label": torch.tensor(n_label, dtype=torch.long),
            "time": torch.tensor(time, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
        }
        return cls(tensors, list(features["case_id"]))

    def __len__(self) -> int:
        return self.tensors["bottleneck"].size(0)

    def batches(self, batch_size: int, device, shuffle: bool):
        order = torch.randperm(len(self)) if shuffle else torch.arange(len(self))
        for start in range(0, len(self), batch_size):
            index = order[start:start + batch_size]
            yield {field: self.tensors[field][index].to(device) for field in self.TENSOR_FIELDS}

    def class_weights(self, field: str, num_classes: int, device) -> torch.Tensor:
        labels = self.tensors[field]
        observed = labels[labels >= 0]
        counts = torch.bincount(observed, minlength=num_classes).float().clamp_min(1.0)
        return (counts.sum() / (num_classes * counts)).to(device)

class ClinicalDataModule:

    def __init__(self, config):
        self.config = config
        self.patients = pd.read_csv(config.csv_path)
        self.encoder = ClinicalEncoder()
        self.train = None
        self.val = None

    def setup(self):
        train_features = torch.load(self.config.train_features_path, map_location="cpu")
        val_features = torch.load(self.config.val_features_path, map_location="cpu")
        train_patients = self.patients[self.patients["PatientID"].isin(train_features["case_id"])]
        self.encoder.fit(train_patients)
        self.train = ClinicalDataset.from_bottlenecks(train_features, self.patients, self.encoder)
        self.val = ClinicalDataset.from_bottlenecks(val_features, self.patients, self.encoder)
        print(f"loaded {len(self.train)} train and {len(self.val)} val features")
        return self

    @property
    def clinical_dim(self) -> int:
        return self.encoder.output_dim

    def survival_bin_edges(self) -> torch.Tensor:
        train_patients = self.patients[self.patients["PatientID"].isin(self.train.case_ids)]
        relapse_times = train_patients[train_patients["Relapse"] == 1]["RFS"].dropna().to_numpy(dtype=float)
        if len(relapse_times) < self.config.n_time_bins:
            relapse_times = train_patients["RFS"].dropna().to_numpy(dtype=float)
        if len(relapse_times) == 0:
            edges = np.linspace(0, 1, self.config.n_time_bins + 1)
        else:
            edges = np.quantile(relapse_times, np.linspace(0, 1, self.config.n_time_bins + 1))
            edges[0], edges[-1] = -np.inf, np.inf
        return torch.tensor(edges, dtype=torch.float32)
