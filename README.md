# HECKTOR 2026 

## Previous approaches
- The three challenge tasks (segmentation, staging, prognosis) are **completely isolated**: no information flows between them.
- The image is downgraded to 96³, without any information about tumor localisation.

## 2026 Guidelines
Participants are invited to develop a multimodal pipeline leveraging FDG PET, CT, and clinical data to:

    Segment primary tumors and lymph nodes
    Infer radiological TN staging
    Predict recurrence-free survival

This unified task reflects a realistic clinical workflow, integrating diagnosis, staging, and prognosis into a single framework.

---

## 2026 Pipeline — Segmentation profonde + têtes-forêts sur embedding

L'entraînement end-to-end conjoint (seg + T/N + survie en un seul forward) était
contraint à `batch_size=2` par la VRAM sur volumes 128³, ce qui rendait les têtes
neuronales de classification et de survie instables. La pipeline est donc
**entièrement découplée**, sans aucune fusion de données :

1. **Phase 1** — segmentation **nnU-Net v2** (seule tâche profonde).
2. **Embedding** — le bottleneck figé de l'encodeur nnU-Net de chaque patient est réduit en un vecteur TEP/CT.
3. **Phase 2** — deux têtes **indépendantes**, sans aucune fusion :
   - un `RandomForest` (stades T et N) entraîné *uniquement* sur l'embedding TEP/CT ;
   - un `RandomSurvivalForest` (survie sans rechute) entraîné *uniquement* sur les
     variables cliniques tabulaires du CSV — l'image n'intervient pas.

### Phase 1 — Segmentation nnU-Net (`train_seg.py`)

La segmentation repose désormais sur **nnU-Net v2**, piloté par le `nnUNetV2Runner` de
MONAI (paquet `nnunetv2`). nnU-Net est auto-configurant : il dérive patch size, batch size,
normalisation par canal et data augmentation de l'empreinte du jeu de données — **plus de
recherche d'hyperparamètres Optuna ni de retrain séparé**.

```
  CT+PET (RAW, déjà prétraités en SUV)
        │
        │ train_seg.py construit l'arborescence brute nnU-Net via liens symboliques :
        │   imagesTr/<case>_0000.nii.gz → CT     (canal 0)
        │   imagesTr/<case>_0001.nii.gz → PET    (canal 1)
        │   labelsTr/<case>.nii.gz      → masque (0=fond, 1=GTVp, 2=GTVn)
        ▼
  nnUNetV2Runner.plan_and_process()   ← empreinte + planification + prétraitement
        │
        │ splits_final.json : fold 0 figé sur split_case_ids (mêmes patients que la
        │ pipeline image — cohérence du split conservée)
        ▼
  nnUNetV2Runner.train_single_model("3d_fullres", fold=0)
        │   perte Dice+CE, deep supervision, SW inference — gérés par nnU-Net
        ▼
  checkpoints nnU-Net + validation/summary.json (Dice par classe)
  → experiments/<exp>/nnunet/nnUNet_results/Dataset001_HECKTOR/...
```

À la première exécution de `train_tn.py`, `ensure_bottlenecks()` (`NNUNetBottleneckExtractor`)
recharge le modèle nnU-Net entraîné, **GÈLE son encodeur**, prétraite chaque patient *via
nnU-Net lui-même* (resampling + normalisation par canal), recadre au patch du plan, et
sauvegarde le **bottleneck du dernier étage d'encodage** vers
`experiments/<exp>/features/{train,val}.pt` (déterministe, réutilisé ensuite).

### Phase 2 — Têtes-forêts indépendantes (`train_tn.py`, `train_survival.py`)

```
   EMBEDDING TEP/CT (image)                  DONNÉES CLINIQUES (CSV)
   features/{train,val}.pt                   HECKTOR_2026_training_data.csv
        │                                          │
        │ pool_embedding : moyenne ⊕ max          │ ClinicalEncoder : âge standardisé +
        │ → embedding (N, 2·C_enc)                 │ one-hot (genre, tabac, alcool,
        │                                          │ perf. status, HPV, traitement)
        ▼                                          ▼
   ┌──────────────┬──────────────┐         RandomSurvivalForest  (train_survival.py)
   ▼              ▼                         cible (event, RFS) → score de risque
 RandomForest T   RandomForest N                   │
 (train_tn.py)    (train_tn.py)                  c-index
 class_weight=balanced                             │
 → stade T        → stade N                    Optuna (val)
   balanced acc     balanced acc
        └──── Optuna (val) ────┘

  • Trois forêts strictement indépendantes, AUCUNE fusion image / clinique.
  • T/N : embedding image seul, lignes au label connu (>= 0).
  • Survie : variables cliniques seules (T/N-stage exclus — ce sont les cibles, non
    disponibles à l'inférence) ; patients à RFS renseigné (> 0).
  • Split train/val partagé avec la pipeline image (même seed) ; modèles en .joblib.
```

### Exécution

```bash
python train_seg.py        # phase 1 : segmentation nnU-Net (auto-configurant, fold 0)
python train_tn.py         # phase 2 : extraction auto des embeddings + RandomForest T/N
python train_survival.py   # phase 2 : RandomSurvivalForest sur données cliniques (CSV)
```

### Variante survie — embedding CT du modèle de fondation CT-FM (`train_foundation_survival.py`)

Approche indépendante des phases ci-dessus : la survie est prédite uniquement à partir
de la **CT**, encodée par le modèle de fondation **CT-FM**
([project-lighter/ct_fm_feature_extractor](https://huggingface.co/project-lighter/ct_fm_feature_extractor)),
un SegResNet pré-entraîné en self-supervised contrastif sur 148 000 scanners (Imaging
Data Commons). Pour chaque patient on extrait un vecteur figé de **512** caractéristiques
(global average pooling du dernier feature map), puis on entraîne un `RandomSurvivalForest`
sur la cible `(événement, RFS)`. Le split train/val et la recherche Optuna (c-index)
restent ceux des autres têtes. L'extraction (GPU) est mise en cache dans
`experiments/<exp>/features/ct_fm_{train,val}.pt` et réutilisée ensuite.

```bash
pip install lighter_zoo               # dépendance du modèle de fondation
python train_foundation_survival.py   # extraction CT-FM (cache) + RandomSurvivalForest
```

---

## Repository structure

```
hecktor2026/
│
├── train_seg.py                 # Phase 1 : segmentation nnU-Net (MONAI nnUNetV2Runner)
├── train_tn.py                  # Phase 2 : RandomForest T et N sur embedding figé
├── train_survival.py            # Phase 2 : RandomSurvivalForest sur données cliniques
├── train_foundation_survival.py # Variante : RandomSurvivalForest sur embedding CT-FM (CT seule)
│
├── src/
│   ├── image_data.py            # split patients + NNUNetBottleneckExtractor + ensure_bottlenecks
│   └── clinical_data.py         # embedding (T/N) + encodeur clinique & survie (RFS)
│
├── utils/
│   └── metrics.py               # balanced_accuracy, c_index
│
└── config.py                    # réglages nnU-Net, split, chemins, n_trials des forêts
```
