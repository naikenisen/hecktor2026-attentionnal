"""Garde-fou contre les caches d'embeddings PÉRIMÉS.

Les extracteurs (CT-FM, bottleneck nnU-Net) sont idempotents : ils sautent l'extraction quand
le fichier de features existe déjà. Ne tester que l'*existence* du fichier est un piège — un
cache généré quand `data_root` contenait moins de patients reste réutilisé indéfiniment, et
les têtes en aval ne voient qu'une fraction de la cohorte. Ici, un cache n'est réputé valide
que s'il couvre EXACTEMENT les patients attendus du split courant ; sinon il est réextrait.
"""
import os
import torch


def cached_case_ids(path: str) -> set:
    """Ensemble des case_id présents dans un fichier de features (.pt)."""
    return set(torch.load(path, map_location="cpu", weights_only=False)["case_id"])


def cache_covers(path: str, expected_ids) -> bool:
    """Vrai si `path` existe et contient exactement les `expected_ids` — les patients
    réellement extractibles du split courant. Faux si absent ou périmé (data_root a changé)."""
    if not os.path.exists(path):
        return False
    return cached_case_ids(path) == set(expected_ids)
