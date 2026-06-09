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

### Phase 1 — Segmentation nnU-Net (paquet `seg`)

La segmentation repose désormais sur **nnU-Net v2**, piloté par le `nnUNetV2Runner` de
MONAI (paquet `nnunetv2`). nnU-Net est auto-configurant : il dérive patch size, batch size,
normalisation par canal et data augmentation de l'empreinte du jeu de données — **plus de
recherche d'hyperparamètres Optuna ni de retrain séparé**.

```
  CT+PET (RAW, déjà prétraités en SUV)
        │
        │ seg.dataset construit l'arborescence brute nnU-Net via liens symboliques :
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

À la première exécution de `tn.train`, `ensure_bottlenecks()` (`NNUNetBottleneckExtractor`)
recharge le modèle nnU-Net entraîné, **GÈLE son encodeur**, prétraite chaque patient *via
nnU-Net lui-même* (resampling + normalisation par canal), recadre au patch du plan, et
sauvegarde le **bottleneck du dernier étage d'encodage** vers
`experiments/<exp>/features/{train,val}.pt` (déterministe, réutilisé ensuite).

### Phase 2 — Têtes-forêts indépendantes (paquets `tn`, `survival`)

```
   EMBEDDING TEP/CT (image)                  DONNÉES CLINIQUES (CSV)
   features/{train,val}.pt                   HECKTOR_2026_training_data.csv
        │                                          │
        │ pool_embedding : moyenne ⊕ max          │ ClinicalEncoder : âge standardisé +
        │ → embedding (N, 2·C_enc)                 │ one-hot (genre, tabac, alcool,
        │                                          │ perf. status, HPV, traitement)
        ▼                                          ▼
   ┌──────────────┬──────────────┐         RandomSurvivalForest  (survival.train)
   ▼              ▼                         cible (event, RFS) → score de risque
 RandomForest T   RandomForest N                   │
   (tn.train)       (tn.train)                    c-index
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

Chaque tâche est un paquet, lancé comme module **depuis la racine du dépôt** (les `.sh`
associés restent à la racine et font `cd` dans le projet avant d'appeler ces commandes) :

```bash
python -m seg.train        # phase 1 : segmentation nnU-Net (auto-configurant, fold 0)
python -m tn.train         # phase 2 : extraction auto des embeddings + RandomForest T/N
python -m survival.train   # phase 2 : RandomSurvivalForest sur données cliniques (CSV)
```

### Variante survie — embedding CT du modèle de fondation CT-FM (`foundation_survival`)

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
pip install lighter_zoo                   # dépendance du modèle de fondation
python -m foundation_survival.train       # extraction CT-FM (cache) + RandomSurvivalForest
```

---

## Repository structure

Le code est séparé en un paquet par tâche (`seg`, `tn`, `survival`, `foundation_survival`),
plus `preprocessing/` (prépa des données, scripts autonomes). Les `.sh` de soumission restent
à la racine. `src/` regroupe le code partagé entre têtes (split, métriques, forêt de survie).

```
hecktor2026/
│
├── seg/                         # Phase 1 — segmentation nnU-Net
│   ├── __init__.py              #   fixe les variables d'env nnU-Net (avant tout import nnunetv2)
│   ├── dataset.py               #   arborescence brute (liens symboliques) + dataset.json + split
│   ├── runner.py                #   pilotage du nnUNetV2Runner (plan/preprocess/train/Dice)
│   └── train.py                 #   point d'entrée  →  python -m seg.train
│
├── tn/                          # Phase 2 — stades T et N (RandomForest sur embedding figé)
│   ├── extractor.py             #   PetCtDataModule + NNUNetBottleneckExtractor + ensure_bottlenecks
│   ├── dataset.py               #   pool_embedding + EmbeddingDataset (embedding ↔ cibles T/N)
│   ├── forest.py                #   RandomForestClassifier + recherche Optuna (balanced acc)
│   └── train.py                 #   point d'entrée  →  python -m tn.train
│
├── survival/                    # Phase 2 — survie RFS (RandomSurvivalForest, données cliniques)
│   ├── dataset.py               #   ClinicalEncoder + ClinicalSurvivalDataset (variables cliniques)
│   └── train.py                 #   point d'entrée  →  python -m survival.train
│
├── foundation_survival/         # Variante — survie RFS sur embedding CT-FM (CT seule)
│   ├── extractor.py             #   extraction des features CT-FM (SegResNet pré-entraîné)
│   ├── dataset.py               #   alignement embedding ↔ cible de survie
│   └── train.py                 #   point d'entrée  →  python -m foundation_survival.train
│
├── preprocessing/               # Prépa des données (scripts autonomes, hors pipeline d'entraînement)
│   ├── convert_nifti_Bq_SUV.py  #   conversion TEP BQML → SUVbw (bq_to_suv)
│   └── preprocess.py            #   resampling / recadrage / normalisation → python preprocessing/preprocess.py
│
├── src/                         # Code partagé entre plusieurs têtes
│   ├── split.py                 #   split_case_ids — split train/val déterministe (toutes les têtes)
│   ├── metrics.py               #   balanced_accuracy, c_index
│   └── survival_forest.py       #   RandomSurvivalForest + Optuna, partagé survival/foundation
│
├── train_seg.sh / train_tn.sh / train_survival.sh / train_foundation_survival.sh / preprocess.sh
└── config.py                    # réglages nnU-Net, split, chemins, n_trials des forêts
```
