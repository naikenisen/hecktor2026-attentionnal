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
3. **Phase 2** — deux têtes **indépendantes**, chacune une forêt aléatoire entraînée
   *uniquement* sur cet embedding : un `RandomForest` pour les stades T et N, un
   `RandomSurvivalForest` pour la survie sans rechute. Aucune variable clinique
   tabulaire n'est fusionnée à l'image.

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

À la première exécution d'une tâche tabulaire, `ensure_bottlenecks()` recharge
best_model.pth, GÈLE le backbone, et extrait le bottleneck spatial de CHAQUE patient
vers experiments/<exp>/features/{train,val}.pt (déterministe, réutilisé ensuite).
```

### Phase 2 — Têtes-forêts sur embedding figé (`train_tn.py`, `train_survival.py`)

```
features/{train,val}.pt   (bottlenecks figés, par case_id)
        │
        │  pool_embedding : GAP (moyenne) ⊕ max global  →  embedding (N, 2·768)
        │  (src/clinical_data.py — joint aux cibles T/N/RFS/event par case_id)
        ▼
   embedding TEP/CT  (aucune variable clinique fusionnée)
        │
        ├──────────────────────────────┬─────────────────────────────┐
        ▼                              ▼                             ▼
  RandomForest T              RandomForest N            RandomSurvivalForest
  (train_tn.py)               (train_tn.py)             (train_survival.py)
  class_weight=balanced       class_weight=balanced     cible (event, RFS)
  → stade T (4 classes)       → stade N (4 classes)     → score de risque
        │                              │                             │
   balanced accuracy            balanced accuracy               c-index
        └──────────── Optuna (val split) ──────────────────────────┘

  • Trois forêts strictement indépendantes, entraînées sur le SEUL embedding image.
  • T/N : lignes au label connu (>= 0) ; survie : patients à RFS renseigné (> 0).
  • Modèles sérialisés en .joblib dans checkpoints/.
```

### Exécution

```bash
python train_seg.py        # phase 1 : recherche HP segmentation
python retrain_seg.py      # phase 1 : retrain final, sauvegarde best_model.pth
python train_tn.py         # phase 2 : extraction auto des embeddings + RandomForest T/N
python train_survival.py   # phase 2 : RandomSurvivalForest (réutilise les embeddings)
```

---

## Repository structure

```
hecktor2026/
│
├── train_seg.py                 # Phase 1 : recherche HP segmentation (Optuna)
├── retrain_seg.py               # Phase 1 : retrain final → best_model.pth
├── train_tn.py                  # Phase 2 : RandomForest T et N sur embedding figé
├── train_survival.py            # Phase 2 : RandomSurvivalForest sur embedding figé
│
├── src/
│   ├── image_data.py            # DataModule seg + BottleneckExtractor + ensure_bottlenecks
│   ├── clinical_data.py         # pool_embedding + EmbeddingDataset (embedding ⊕ cibles T/N/survie)
│   └── networks.py              # SwinUNETRBackbone : returns (seg_logits, bottleneck)
│
├── utils/
│   ├── losses.py                # seg_loss = DiceFocal
│   └── metrics.py               # balanced_accuracy, c_index
│
└── config.py                    # feature_size, lr, batch_size, n_trials des forêts
```
