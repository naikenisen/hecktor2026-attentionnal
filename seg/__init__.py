"""Phase 1 — segmentation nnU-Net v2 (MONAI `nnUNetV2Runner` pilotant `nnunetv2`).

nnU-Net lit `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` dans l'environnement
dès l'import de `nnunetv2`. On les fixe ici, à l'import du paquet `seg` — donc avant tout
import d'un sous-module qui charge `nnunetv2` (`seg.dataset`, `seg.runner`).
"""
import os
import config

os.environ["nnUNet_raw"] = config.nnunet_raw_dir
os.environ["nnUNet_preprocessed"] = config.nnunet_preprocessed_dir
os.environ["nnUNet_results"] = config.nnunet_results_dir
