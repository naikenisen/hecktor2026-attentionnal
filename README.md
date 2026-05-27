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

## 2026 Pipeline — Decoupled Two-Phase Multitask Learning

L'entraînement end-to-end conjoint (seg + T/N + survie en un seul forward) était
contraint à `batch_size=2` par la VRAM du SwinUNETR sur volumes 128³. Ce batch
minuscule rendait la survie (DeepHit, qui a besoin de paires d'événements) et la
classification (forte sur-représentation de N2) incapables d'apprendre, pendant que
la pondération par incertitude étouffait ces tâches difficiles. La solution :
**découpler la segmentation des tâches tabulaires** en deux phases.

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

Après convergence, train_seg.py recharge best_model.pth, GÈLE le backbone, et
extrait le bottleneck spatial de CHAQUE patient (+ cibles T/N/RFS/event/clinique)
vers experiments/<exp>/features/{train,val}.pt  (~150 MB, déterministe).
```

### Phase 2 — Têtes cliniques sur features cachées (`train_clinical.py`)

```
features/{train,val}.pt   (bottlenecks figés + cibles tabulaires)
        │
        │  gros batch (64) désormais possible : plus de volumes 3D en mémoire
        ▼
ClinicalModel (src/clinical_model.py)
        │
        ├─ bottleneck ─┬─► T-Head (GAP→256→logits B,4) ─► L_T = CrossEnt pondérée
        │              └─► N-Head (GAP→256→logits B,4) ─► L_N = CrossEnt pondérée
        │                         │
        │              t_feat + n_feat → Linear(512, d_model) → token_tn (B,1,d)
        │                         │
        │   Clinical (B, 22) ─ MLP one-hot (src/clinical_encoder.py) ─► token_clin (B,1,d)
        │                         │
        ▼                         ▼
   ┌──────────────────────────────────────────────┐
   │           Cross-Attention Fusion              │
   │           (src/cross_attention.py)            │
   │  Q = cat([CLS, token_clin, token_tn]) (B,3,d) │
   │  K = V = Linear(768, d)(bottleneck) (B,64,d)  │
   │  attn(Q,K,V) → enriched CLS (B, d_model)      │
   └─────────────────────┬─────────────────────────┘
                         ▼
            Survival Head (Discrete-Time, src/heads.py)
            Linear(d → 256 → T) ; T intervalles = quantiles
            des temps d'événement du train
                         │
            L_Surv = DeepHit (utils/losses.py)
            calculée UNIQUEMENT sur les patients RFS>0 (survie nettoyée)

═══════════════════════════════════════════════════════════════════════════════

LOSS PHASE 2 :  L = w_T·L_T + w_N·L_N + w_Surv·L_Surv
  • Poids dynamiques par Uncertainty Weighting (Kendall et al.) sur 3 tâches
  • CrossEntropy T/N pondérées par l'inverse des fréquences de classe
  • Variables cliniques catégorielles encodées en one-hot (dim 22)
  • Le backbone reste figé : aucun gradient ne remonte vers le SwinUNETR

═══════════════════════════════════════════════════════════════════════════════
```

### Exécution

```bash
python train_seg.py        # phase 1 : seg + extraction auto des bottlenecks
python train_clinical.py   # phase 2 : T/N + survie
# (train.sh enchaîne les deux ; train_seg.py --extract-only ré-extrait sans réentraîner)
```

---

## Repository structure

```
hecktor2026/
│
├── train_seg.py                 # Phase 1 : segmentation seule + extraction des bottlenecks
├── train_clinical.py            # Phase 2 : têtes T/N + survie sur features cachées
│
├── src/
│   ├── dataset.py               # HECKTORDataset + one-hot clinique + missing values
│   ├── transforms.py            # MONAI augmentations (flip, noise, intensity)
│   ├── preprocessing.py         # resampling 2×2×2 mm³, crop 128³
│   │
│   ├── swinunetr.py             # SwinUNETRBackbone (MONAI subclass)
│   │                            # returns (seg_mask, bottleneck) — backbone phase 1
│   ├── clinical_model.py        # ClinicalModel : bottleneck → logits T/N + survie (phase 2)
│   ├── heads.py                 # TNHead (returns feat + logits) + SurvivalHead
│   ├── cross_attention.py       # CrossAttentionFusion
│   └── clinical_encoder.py     # Clinical MLP (one-hot → 64 → d_model)
│
├── utils/
│   ├── losses.py                # DiceFocal + CrossEntropy + DeepHit
│   └── metrics.py               # C-index, Dice, Balanced Accuracy
│
└── config.py                    # d_model, T, N_warmup, lr, batch_size
```
