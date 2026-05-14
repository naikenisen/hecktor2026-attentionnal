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

## 2026 Pipeline — End-to-End Multitask Learning

```
  CT+PET (RAW)
        |
        | preprocessing (src/preprocessing.py)
        | resampling 2×2×2 mm³, crop 128³ centred on tumour
        |
CT+PET (B, 2, 128, 128, 128)
        |
        | dataloading and transformation (src/dataloader.py, src/transforms.py)
        | missing values: categorical variables → "Unknown" class
        | continuous variables → normalisation + median imputation
        │
        ▼
    SwinUNETR (src/swinunetr.py)
    SSL pre-trained weights on 5050 CTs (model_swinvit.pt)
        │
    L_Seg = Dice+Focal (utils/losses.py)
        │
        ├──► seg_mask (B, 3, D, H, W)
        │
        └──► bottleneck (B, C, D', H', W')
                  │
        ┌─────────┼─────────┐
        │                   │
        ▼                   ▼
T-Head (src/heads.py)    N-Head (src/heads.py)
  GAP → hidden (B,256)    GAP → hidden (B,256)
      → logits (B,4)          → logits (B,4)
        │                   │
   L_T = CrossEnt      L_N = CrossEnt  (utils/losses.py)
        │                   │
        └─────────┬─────────┘
                  │
                  │  t_feat (B,256) + n_feat (B,256)
                  │  → nn.Linear(512, d_model)
                  │  → token_tn (B, 1, d_model)
                  │
                  │          Clinical (B, 7)
                  │               │
                  │     MLP (7→64→d_model) (src/clinical_encoder.py)
                  │     NaN → "Unknown" class for categoricals
                  │     NaN → median imputation for continuous
                  │               │
                  │          token_clin (B, 1, d_model)
                  │               │
                  ▼               ▼
      ┌─────────────────────────────────────────────────┐
      │              Cross-Attention Fusion              │
      │              (src/cross_attention.py)            │
      │                                                  │
      │  Q (B, 3, d_model) :                             │
      │  cat([CLS, token_clin, token_tn], dim=1)         │
      │                                                  │
      │  K = V (B, N, d_model) :                         │
      │  Linear(C, d_model)(bottleneck.flatten(2)        │
      │  .permute(0,2,1))                                │
      │                                                  │
      │  attn(Q, K, V) → enriched CLS (B, d_model)       │
      └─────────────────────────┬───────────────────────┘
                                │
                                ▼
                Survival Head (Discrete-Time)
                nn.Linear(d_model → 256 → T) (src/heads.py)
                T intervals defined by quantiles
                over event times from the train set
                                │
                    ┌───────────┴────────────┐
                    │                        │
                    ▼ (training)             ▼ (inference)
             raw logits (B, T)          softmax(logits)
             L_Surv = DeepHit               │
             (utils/losses.py)         Risk Probabilities (B, T)
             gradient clipping
             max_norm = 1.0

═══════════════════════════════════════════════════════════════════════════════

TOTAL LOSS FUNCTION (End-to-End):

  L_Total = w₁·L_Seg + w₂·L_T + w₃·L_N + w₄·L_Surv

  • Dynamic weights (wᵢ) adjusted by Uncertainty Weighting (Kendall et al.)
  • Gradients from all losses backpropagate through the Bottleneck
  • Backpropagation of L_Surv via cross-attention → Bottleneck
  • Warm-up: T/N and Survival heads are frozen for N_warmup epochs
    → only segmentation is trained
    → then progressive unfreezing of all heads

═══════════════════════════════════════════════════════════════════════════════
```

---

## Repository structure

```
hecktor2026/
│
├── train.py                     # single entry point
│
├── src/
│   ├── dataset.py               # HECKTORDataset + missing value handling
│   ├── transforms.py            # MONAI augmentations (flip, noise, intensity)
│   ├── preprocessing.py         # resampling 2×2×2 mm³, crop 128³
│   │
│   ├── model.py                 # MultitaskModel — central file
│   │                            # forward() connects all components
│   │                            # guarantees end-to-end backprop
│   │
│   ├── swinunetr.py             # SwinUNETRMultitask (MONAI subclass)
│   │                            # returns (seg_mask, bottleneck)
│   ├── heads.py                 # TNHead (returns feat + logits) + SurvivalHead
│   ├── cross_attention.py       # CrossAttentionFusion
│   └── clinical_encoder.py     # Clinical MLP (7 → 64 → d_model)
│
├── utils/
│   ├── losses.py                # DiceFocal + CrossEntropy + DeepHit
│   └── metrics.py               # C-index, Dice, Balanced Accuracy
│
└── config.py                    # d_model, T, N_warmup, lr, batch_size
```
