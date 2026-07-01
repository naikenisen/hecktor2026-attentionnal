"""Split train/test matérialisé sur le disque — source unique pour seg.

Le découpage n'est plus calculé en code : il EST donné par l'arborescence
`data_root/{train,test}/<PatientID>/…` (générée depuis la colonne `split` du CSV).
"""
import os

TRAIN = "train"
TEST = "test"
SPLITS = (TRAIN, TEST)


def split_dir(config, split: str) -> str:
    """Chemin du sous-dossier d'un split ("train" ou "test") sous `data_root`."""
    return os.path.join(config.data_root, split)


def patient_dir(config, split: str, case_id: str) -> str:
    """Chemin du dossier d'un patient dans son split."""
    return os.path.join(config.data_root, split, case_id)


def case_ids(config, split: str) -> list:
    """case_ids (sous-dossiers patients) d'un split, triés."""
    d = split_dir(config, split)
    return sorted(
        name for name in os.listdir(d)
        if os.path.isdir(os.path.join(d, name))
    )
