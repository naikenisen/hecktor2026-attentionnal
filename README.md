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
contraint à `batch_size=2` par la VRAM du SwinUNETR sur volumes 128³, ce qui rendait
les têtes neuronales de classification et de survie instables. La pipeline est donc
**entièrement découplée**, sans aucune fusion de données :

1. **Phase 1** — segmentation SwinUNETR (seule tâche profonde).
2. **Embedding** — le bottleneck figé de chaque patient est réduit en un vecteur TEP/CT.
3. **Phase 2** — deux têtes **indépendantes**, sans aucune fusion :
   - un `RandomForest` (stades T et N) entraîné *uniquement* sur l'embedding TEP/CT ;
   - un `RandomSurvivalForest` (survie sans rechute) entraîné *uniquement* sur les
     variables cliniques tabulaires du CSV — l'image n'intervient pas.

### Phase 1 — Segmentation (`train_seg.py`)

```
  CT+PET (RAW)
        |
        | preprocessing : resampling 2×2×2 mm³, crop 128³ centré sur la tumeur
        | transforms (src/transforms.py) : flip, noise, intensity
        ▼
CT+PET (B, 2, 128, 128, 128)
        ▼
    SwinUNETR (src/swinunetr.py)
    Poids SSL pré-entraînés sur 5050 CTs (model_swinvit.pt)
        │
    L_Seg = Dice+Focal (utils/losses.py)  ← SEULE perte de la phase 1
        │
        ├──► seg_mask (B, 3, D, H, W)        → sélection du best model (Dice)
        │
        └──► bottleneck (B, 768, 4, 4, 4)

À la première exécution de train_tn.py, `ensure_bottlenecks()` recharge best_model.pth,
GÈLE le backbone, et extrait le bottleneck spatial de CHAQUE patient vers
experiments/<exp>/features/{train,val}.pt (déterministe, réutilisé ensuite).
```

### Phase 2 — Têtes-forêts indépendantes (`train_tn.py`, `train_survival.py`)

```
   EMBEDDING TEP/CT (image)                  DONNÉES CLINIQUES (CSV)
   features/{train,val}.pt                   HECKTOR_2026_training_data.csv
        │                                          │
        │ pool_embedding : moyenne ⊕ max          │ ClinicalEncoder : âge standardisé +
        │ → embedding (N, 2·768)                   │ one-hot (genre, tabac, alcool,
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
python train_seg.py        # phase 1 : recherche HP segmentation
python retrain_seg.py      # phase 1 : retrain final, sauvegarde best_model.pth
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
├── train_seg.py                 # Phase 1 : recherche HP segmentation (Optuna)
├── retrain_seg.py               # Phase 1 : retrain final → best_model.pth
├── train_tn.py                  # Phase 2 : RandomForest T et N sur embedding figé
├── train_survival.py            # Phase 2 : RandomSurvivalForest sur données cliniques
├── train_foundation_survival.py # Variante : RandomSurvivalForest sur embedding CT-FM (CT seule)
│
├── src/
│   ├── image_data.py            # DataModule seg + BottleneckExtractor + ensure_bottlenecks
│   ├── clinical_data.py         # embedding (T/N) + encodeur clinique & survie (RFS)
│   └── networks.py              # SwinUNETRBackbone : returns (seg_logits, bottleneck)
│
├── utils/
│   ├── losses.py                # seg_loss = DiceFocal
│   └── metrics.py               # balanced_accuracy, c_index
│
└── config.py                    # feature_size, lr, batch_size, n_trials des forêts
```
