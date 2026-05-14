# HECKTOR 2026 — Analyse comparative des pipelines de prédiction de survie

Ce document présente trois approches pour la tâche de prédiction de la survie sans récidive (RFS) dans le challenge HECKTOR : la pipeline initiale (point de départ), la solution gagnante de HECKTOR 2025, et une proposition pour HECKTOR 2026.

---

## Glossaire des termes techniques

**Backbone** : réseau de neurones principal qui extrait des features à partir des données d'entrée. C'est le "corps" du modèle, par opposition aux "têtes" (small networks ajoutés à la fin pour produire des prédictions spécifiques). Par exemple, un ResNet-18 utilisé pour extraire des features d'images est un backbone.

**Branche** : un chemin dans l'architecture du réseau, dédié à une tâche spécifique. Une branche peut contenir un backbone complet (3D U-Net, DenseNet) ou simplement quelques couches FC. Le terme désigne la structure du graphe computationnel, pas un type de réseau particulier.

**Deep features** : vecteur numérique produit par les couches intermédiaires d'un réseau de neurones, par opposition aux features handcrafted (volume tumoral, SUVmax). Ces dimensions n'ont pas d'interprétation individuelle, mais collectivement elles encodent une représentation compressée apprise par le réseau pour résoudre une tâche.

**MLP (Multi-Layer Perceptron)** : empilement de couches linéaires (fully connected) séparées par des activations non-linéaires (ReLU). Utilisé pour traiter des vecteurs (données tabulaires, features déjà extraites), pas des images.

**End-to-end** : tous les composants du modèle sont entraînés conjointement avec une seule loss totale. Un seul `loss.backward()` propage les gradients à travers tout le graphe computationnel. Opposé : approche en plusieurs étapes où chaque composant est entraîné séparément.

**Multitâche** : le réseau apprend à résoudre plusieurs tâches simultanément (segmentation + classification + survie). La loss totale est la somme pondérée des losses individuelles. Les tâches auxiliaires servent de régularisateurs et partagent des représentations communes.

**Cross-attention** : mécanisme d'attention où des tokens issus de sources différentes interagissent. Chaque token est projeté en queries, keys, values dans un espace commun, et le mécanisme apprend à pondérer dynamiquement leurs interactions. Permet une fusion plus expressive qu'une simple concaténation.

**Token** : vecteur numérique constituant l'unité d'entrée d'un module Transformer. Dans une fusion multimodale, chaque modalité (image, clinique, staging) est représentée par un token.

---

## 1. Pipeline initiale (point de départ)

### Architecture

Deux extracteurs de features parallèles, fusionnés par un MLP, suivis d'un modèle de survie classique en post-hoc.

### Framework
- **MONAI** pour le pipeline d'imagerie
- **icare** (librairie externe) pour le modèle de survie final

### Composants

| Composant | Type | Entrée | Sortie | Dimensions |
|-----------|------|--------|--------|------------|
| Extracteur d'images | ResNet-18 3D (FC → Identity) | CT+PET (2 canaux, 96³) | f_img | 512 |
| MLP clinique | MLP (linear → BN → Dropout → ReLU) | Features cliniques (clin_dim) | f_clin | 32 |
| MLP de fusion | MLP (544 → 512 → 256 → 128) | f_img + f_clin concaténés | features | 128 |
| Risk head | Linéaire | features | risk_score | 1 |
| Modèle final | BaggedIcareSurvival (icare) | features | risk_score RFS | 1 |

### Losses (entraînement du FusedFeatureExtractor uniquement)

- **DeepHitLoss** appliquée sur risk_score : likelihood Cox + ranking
- **SurvivalContrastiveLoss** appliquée sur les features 128 dims : margin=2.0, temperature=0.1
- Loss totale : `L = DeepHitLoss + 0.1 × ContrastiveLoss`

### Mode d'entraînement (en deux étapes)

1. **Étape neuronale** : le FusedFeatureExtractor est entraîné sur DeepHitLoss + ContrastiveLoss pendant `feature_epochs_per_iteration` époques.
2. **Étape classique** : BaggedIcareSurvival est entraîné sur les features 128 dims extraites par le réseau (modèle non différentiable, pas de backprop).
3. Itération alternée des deux étapes (par défaut 25 cycles × 10 époques = 250 époques effectives).

### Limites identifiées
- Les trois tâches du challenge (segmentation, staging, pronostic) sont **complètement isolées** : aucune information ne circule entre elles.
- L'image est dégradée à 96³, sans information sur la localisation tumorale.
- Pas d'utilisation des masques de segmentation pour le pronostic.

---

## 2. Solution gagnante HECKTOR 2025 (équipe SIMS-LIFE)

### Architecture

Version modifiée de **DeepMTS** : trois branches connectées en cascade, partageant des informations via leurs deep features.

### Framework
- **TensorFlow** (1.x et 2.x selon les setups)
- Modèle entraîné de zéro, sans poids pré-entraînés

### Composants

| Branche | Type de réseau | Entrée | Sortie principale | Deep features extraites |
|---------|----------------|--------|-------------------|-------------------------|
| Segmentation | 3D U-Net modifié (blocs résiduels) | CT+PET concaténés (2 canaux, 128³) | Masque tumoral | z₁ (124 dims) depuis l'encodeur |
| Classification HPV | 3D DenseNet modifié (blocs denses) | CT+PET + carte de proba tumorale (3 canaux) | Prédiction HPV (binaire) | z₂ (112 dims) |
| Survie | 3 couches FC | z₁ (124) + z₂ (112) + clinique (7) = 243 dims | Score de risque RFS | — |

### Connexions entre branches

1. La branche segmentation produit le masque tumoral.
2. Le masque (carte de probabilité) est concaténé avec CT+PET et envoyé dans la branche classification.
3. Les deep features z₁ et z₂ sont concaténées avec les 7 features cliniques brutes et envoyées dans la branche survie.

### Losses

```
L_Total = L_Seg + L_Class + L_Surv + 0.1 × L2_Reg
```

| Loss | Branche | Formulation |
|------|---------|-------------|
| L_Seg | Segmentation | Dice loss + Focal loss (α=0.25, γ=2) |
| L_Class | Classification HPV | Binary focal cross-entropy |
| L_Surv | Survie | Cox negative log partial likelihood |
| L2_Reg | Survie (FC layers) | Régularisation L2 |

### Mode d'entraînement (end-to-end multitâche)

**Un seul forward pass** :
1. CT+PET entrent dans la branche segmentation → masque + z₁
2. CT+PET + masque entrent dans la branche classification → HPV + z₂
3. z₁ + z₂ + clinique entrent dans la branche survie → risk_score

**Calcul de la loss totale** : somme pondérée des 4 losses individuelles, calculées en parallèle sur leurs prédictions respectives.

**Un seul backward pass** :
- L_Seg propage des gradients dans la branche segmentation uniquement.
- L_Class propage des gradients dans la branche classification (et indirectement dans segmentation via la carte de probabilité).
- **L_Surv propage des gradients dans la branche survie, puis via z₁ dans la branche segmentation, et via z₂ dans la branche classification.** C'est ce qui force les deux backbones à produire des features informatives pour le pronostic.

**Gestion des labels manquants** :
- Patients sans label HPV : forward pass complet, mais exclusion de L_Class du backward.
- Patients sans masque de segmentation : forward pass complet, mais exclusion de L_Seg du backward.
- Le réseau apprend quand même des images via L_Surv.

### Résultats
- C-index validation HECKTOR : **0.583**
- 1ʳᵉ place sur la tâche de survie HECKTOR 2025

---

## 3. Proposition pour HECKTOR 2026

### Architecture

Framework multitâche end-to-end à **quatre branches** (segmentation, classification HPV, staging T/N, survie), avec **fusion par cross-attention** au lieu de concaténation.

### Framework et poids
- **MONAI** pour SwinUNETR et l'infrastructure
- **AICONSlab/3DINO** (projet externe, non natif MONAI) pour le ViT 3D pré-entraîné en self-supervised sur ~100 000 volumes médicaux multi-modalités
- PyTorch pour la cross-attention custom

### Composants

| Branche | Type de réseau | Pré-entraînement | Entrée | Sortie principale | Deep features |
|---------|----------------|------------------|--------|-------------------|---------------|
| Segmentation | SwinUNETR (MONAI) | Swin self-supervised (5050 CTs) | CT+PET (2 canaux, 128³) | Masque tumoral | z₁ depuis l'encodeur |
| Classification | ViT 3D (3DINO) | 3DINO self-supervised (100K volumes) | CT+PET + masque (3 canaux) | HPV + T-stage + N-stage | z₂ depuis le CLS token |
| MLP clinique | MLP (7 → 64 → 32) | Aucun | 7 features cliniques | Vecteur clinique enrichi | f_clin (32 dims) |
| Cross-attention fusion | Transformer encoder (2 layers, 8 heads) | Aucun | 4 tokens projetés en d_model | CLS token | 256 dims |
| Survie | FC layers (256 → 128 → 1) | Aucun | CLS token de la fusion | Score de risque RFS | — |

### Têtes de classification sur le ViT

Le ViT 3D produit z₂, sur lequel sont branchées **trois têtes parallèles** :
- Tête HPV : Linear(d_vit → 2)
- Tête T-stage : Linear(d_vit → 4)
- Tête N-stage : Linear(d_vit → 4)

### Module de cross-attention (cœur de l'innovation)

Quatre tokens en entrée, chacun projeté vers une dimension commune d_model=256 :
1. **Token image-seg** : projection de z₁
2. **Token image-class** : projection de z₂
3. **Token staging** : projection de la concaténation des logits T + N (8 dims)
4. **Token clinique** : projection de f_clin (32 dims)

Un **CLS token apprenable** est ajouté en première position. Les 5 tokens passent dans un Transformer encoder (self-attention multi-têtes), qui apprend à pondérer dynamiquement les interactions entre modalités. Le CLS token de sortie agrège l'information de tous les autres tokens via attention.

### Losses

```
L_Total = L_Seg + L_HPV + L_T + L_N + L_Surv + λ × L2_Reg
```

| Loss | Branche | Formulation |
|------|---------|-------------|
| L_Seg | Segmentation | Dice + Focal |
| L_HPV | Classification | Focal cross-entropy |
| L_T | Tête T-stage | CrossEntropy (4 classes) |
| L_N | Tête N-stage | CrossEntropy (4 classes) |
| L_Surv | Survie | Cox negative log partial likelihood |
| L2_Reg | FC survie | Régularisation L2 |

### Mode d'entraînement

**Forward pass complet** :
1. SwinUNETR(CT+PET) → masque + z₁
2. ViT_3DINO(CT+PET+masque) → z₂ → têtes HPV/T/N
3. MLP_clinique(features cliniques) → f_clin
4. CrossAttention(projection(z₁), projection(z₂), projection(T+N logits), projection(f_clin), CLS) → CLS_out
5. FC_survie(CLS_out) → risk_score

**Backward pass** : L_Total.backward() propage les gradients à travers tout le graphe. En particulier, L_Surv remonte via le CLS token, se distribue sur les 4 tokens via les poids d'attention apprises, puis remonte dans les backbones SwinUNETR et ViT, dans les têtes T/N, et dans le MLP clinique. Chaque composant est ainsi optimisé conjointement par toutes les losses auxquelles il contribue.

**Gestion des labels manquants** : même stratégie que les gagnants 2025 — exclusion des losses concernées pour les patients sans label, forward pass complet maintenu.

### Avantages attendus
- **SwinUNETR** : meilleure segmentation que le 3D U-Net basique grâce à l'attention par fenêtres.
- **ViT 3DINO pré-entraîné** : démarrage avec des features pré-apprises sur 100K volumes médicaux, là où DeepMTS partait de zéro.
- **Cross-attention** : fusion adaptative au patient, ignorant les modalités redondantes et amplifiant les complémentaires.
- **Branche T/N supplémentaire** : exploitation d'une supervision additionnelle non utilisée par les gagnants 2025.

---

## Schéma comparatif des trois approches

### Pipeline initiale

```
CT+PET (2 ch, 96³)              Clinical (clin_dim)
        │                                │
        ▼                                ▼
   ResNet-18 3D                    MLP clinique
   (FC → Identity)              (64→32, BN, Dropout)
        │                                │
        ▼                                ▼
   f_img (512)                      f_clin (32)
        │                                │
        └──────────► Concat ◄────────────┘
                       │
                  544 dims
                       │
                       ▼
                  MLP fusion
              (544→512→256→128)
                       │
                       ▼
                features (128)
                  │        │
                  ▼        ▼
            Risk head   ┌─ BaggedIcareSurvival
            (128→1)     │  (icare, post-hoc,
                  │     │   non différentiable)
                  ▼     │
        DeepHit+Contrast│
              loss      ▼
                  risk_score final
```

### Solution gagnante HECKTOR 2025 (SIMS-LIFE)

```
CT+PET (2 ch, 128³)
        │
        ▼
┌───────────────────┐
│ 3D U-Net          │──► Masque seg ──► L_Seg (Dice+Focal)
│ (blocs résiduels) │
│                   │──► z₁ (124 dims) ──┐
└───────────────────┘                    │
                                         │
CT+PET + masque (3 ch)                   │
        │                                │
        ▼                                │
┌───────────────────┐                    │
│ 3D DenseNet       │──► HPV pred ──► L_Class (Focal CE)
│ (blocs denses)    │                    │
│                   │──► z₂ (112 dims) ──┤
└───────────────────┘                    │
                                         │
Clinical (7 dims) ───────────────────────┤
                                         │
                                         ▼
                                  Concat (243 dims)
                                         │
                                         ▼
                                  Branche survie
                                  (3 FC layers)
                                         │
                                         ▼
                                  risk_score ──► L_Surv (Cox)

L_Total = L_Seg + L_Class + L_Surv + 0.1×L2_Reg
Entraînement end-to-end, gradients remontent partout
```

### Proposition pour HECKTOR 2026

```
CT+PET (2 ch, 128³)
        │
        ▼
┌────────────────────────┐
│ SwinUNETR (MONAI)      │──► Masque seg ──► L_Seg (Dice+Focal)
│ Poids pré-entraînés    │
│ (Swin SSL, 5050 CTs)   │──► z₁ ──┐
└────────────────────────┘         │
                                   │
CT+PET + masque (3 ch)             │
        │                          │
        ▼                          │
┌────────────────────────┐         │
│ ViT 3D (AICONSlab)     │         │
│ Poids pré-entraînés    │──► z₂ ──┤
│ 3DINO SSL (100K vol.)  │         │
│                        │         │
│ + 3 têtes parallèles : │         │
│   - HPV head ──────────┼──► L_HPV (Focal CE)
│   - T-head ────────────┼──► L_T  (CE 4 classes)
│   - N-head ────────────┼──► L_N  (CE 4 classes)
│                        │         │
│   T-logits, N-logits ──┼─────────┤
└────────────────────────┘         │
                                   │
Clinical (7 dims)                  │
        │                          │
        ▼                          │
   MLP clinique                    │
   (7 → 64 → 32) ──► f_clin ───────┤
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │  Projections        │
                          │  z₁, z₂, T+N logits,│
                          │  f_clin → d_model   │
                          │  + CLS token        │
                          └─────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │  Cross-Attention    │
                          │  Transformer (2L,   │
                          │  8 heads)           │
                          └─────────────────────┘
                                   │
                                   ▼
                            CLS token (256)
                                   │
                                   ▼
                            FC survie
                            (256→128→1)
                                   │
                                   ▼
                            risk_score ──► L_Surv (Cox)

L_Total = L_Seg + L_HPV + L_T + L_N + L_Surv + λ×L2_Reg
Entraînement end-to-end, gradients de L_Surv remontent
jusqu'aux backbones via la cross-attention
```

---

## Tableau récapitulatif

| | Pipeline initiale | Gagnants 2025 | Proposition 2026 |
|---|---|---|---|
| **Backbone segmentation** | — (séparé) | 3D U-Net (blocs résiduels) | SwinUNETR (MONAI) |
| **Backbone classification** | ResNet-18 3D | 3D DenseNet (blocs denses) | ViT 3D (3DINO) |
| **Poids pré-entraînés** | Aucun | Aucun | Swin SSL + 3DINO SSL |
| **Framework** | MONAI + icare | TensorFlow | MONAI + PyTorch + AICONSlab |
| **Tâches couplées** | Aucune | Seg + HPV + Survie | Seg + HPV + T + N + Survie |
| **Fusion multimodale** | Concaténation + MLP | Concaténation | Cross-attention |
| **Données cliniques** | MLP (32 dims) | Brutes (7 dims) | MLP (32 dims) → token |
| **Résolution image** | 96³ | 128³ | 128³ |
| **Mode d'entraînement** | 2 étapes alternées | End-to-end multitâche | End-to-end multitâche |
| **Modèle final survie** | BaggedIcareSurvival | 3 FC layers | FC layers (256→128→1) |
| **Loss survie** | DeepHit + Contrastive | Cox NLPL | Cox NLPL |