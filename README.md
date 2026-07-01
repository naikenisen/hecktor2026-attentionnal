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
        │ splits_final.json : fold 0 figé sur le split train/test du disque (mêmes patients
        │ que la pipeline image — cohérence du split conservée)
        ▼
  nnUNetV2Runner.train_single_model("3d_fullres", fold=0)
        │   perte Dice+CE, deep supervision, SW inference — gérés par nnU-Net
        ▼
  checkpoints nnU-Net + validation/summary.json (Dice par classe)
  → results/nnUNet_results/Dataset001_HECKTOR/...
```

L'extraction du bottleneck fait l'objet d'un **script séparé**, `seg.extract`
(`python -m seg.extract`), à lancer sur le cluster après la segmentation. Il recharge le
modèle nnU-Net entraîné, **GÈLE son encodeur**, prétraite chaque patient *via nnU-Net
lui-même* (resampling + normalisation par canal), recadre au patch du plan, réduit le
**bottleneck du dernier étage d'encodage** en un vecteur par patient (`pool_embedding` :
moyenne ⊕ max) et écrit un CSV **unique** `tables/bottleneck.csv` (`PatientID` + `feat_i`,
tous splits confondus). Ce CSV est ensuite rapatrié dans `tables/` et lu tel quel par les
têtes en aval — **plus aucune extraction ailleurs dans le dépôt**.

### Phase 2 — Têtes-forêts indépendantes (paquets `tn`, `clinical_survival`)

```
   EMBEDDING TEP/CT (image)                  DONNÉES CLINIQUES (CSV)
   tables/bottleneck.csv                     tables/HECKTOR_2026_training_data.csv
        │                                          │
        │ features poolées (produites par          │ ClinicalEncoder : âge standardisé +
        │ seg.extract) : PatientID + feat_i        │ one-hot (genre, tabac, alcool,
        │ split via colonne `split` du CSV clinique│ perf. status, HPV, traitement)
        ▼                                          ▼
   ┌──────────────┬──────────────┐         RandomSurvivalForest  (clinical_survival.train)
   ▼              ▼                         cible (event, RFS) → score de risque
 RandomForest T   RandomForest N                   │
   (tn.train)       (tn.train)                    c-index
 class_weight=balanced                             │
 → stade T        → stade N                  GridSearchCV (CV)
   balanced acc     balanced acc
        └─ GridSearchCV (test) ─┘

  • Trois forêts strictement indépendantes, AUCUNE fusion image / clinique.
  • T/N : embedding image seul, lignes au label connu (>= 0).
  • Survie : variables cliniques seules (T/N-stage exclus — ce sont les cibles, non
    disponibles à l'inférence) ; patients à RFS renseigné (> 0).
  • Split train/test partagé avec la pipeline image (arborescence du disque).
  • Aucun modèle de forêt n'est sauvegardé : seuls les scores (balanced acc, c-index)
    sont affichés ; le seul artefact lourd d'un run est le modèle nnU-Net (results/).
```

### Exécution

Chaque tâche est un paquet, lancé comme module **depuis la racine du dépôt** (les scripts
Slurm de `slurm/` font `cd` dans le projet avant d'appeler ces commandes) :

```bash
python -m seg.train               # phase 1   : segmentation nnU-Net (auto-configurant, fold 0)
python -m seg.extract             # phase 1bis : bottleneck → tables/bottleneck.csv (sur le cluster)
python -m tn.train                # phase 2   : RandomForest T/N (lit tables/bottleneck.csv)
python -m clinical_survival.train # phase 2   : RandomSurvivalForest sur données cliniques (CSV)
python -m nnunet_survival.train   # phase 2   : RandomSurvivalForest sur le bottleneck nnU-Net
```

`seg.extract` tourne sur le cluster (GPU) et produit `tables/bottleneck.csv` ; ce fichier est
ensuite rapatrié dans `tables/`. Les têtes de phase 2 ne lisent que `tables/bottleneck.csv` et
`tables/HECKTOR_2026_training_data.csv` — elles n'extraient rien.

### Sorties d'un run (`results/`)

Tout ce que produit un run est centralisé dans `results/` (à la racine du dépôt), sans
sous-dossier d'expérience ni versionnage : les fichiers sont **écrasés à chaque exécution**.
L'idée : archiver puis supprimer `results/` après chaque run pour garder une trace exacte de
ce qui a été produit.

```
results/
├── datalist.json                # trace du split fourni à nnU-Net
└── nnUNet_results/              # modèle nnU-Net entraîné (seul sous-dossier, imposé par nnU-Net)
    └── Dataset001_HECKTOR/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/…
```

Les features du bottleneck ne vont **pas** dans `results/` mais dans `tables/bottleneck.csv`
(produit par `seg.extract`, versionné avec les autres tables).

Restent **hors de `results/`**, sur le scratch à côté des données (`dataset/`), car lourds ou
réutilisables : `nnUNet_raw/` (liens symboliques), `nnUNet_preprocessed/` (données prétraitées
par nnU-Net) et `clinical_clean.csv`. **Aucun modèle de forêt (RF/RSF) n'est sauvegardé** :
seuls les scores (balanced accuracy, c-index) sont affichés.

### Variante survie — bottleneck nnU-Net (`nnunet_survival`)

Troisième tête de survie : on réutilise **le même embedding que `tn`** — le bottleneck de
l'encodeur nnU-Net (`src.nnunet_embedding`, extrait une seule fois puis mis en cache dans
`results/{train,test}.csv`) — mais on entraîne un `RandomSurvivalForest` sur la cible
`(événement, RFS)` au lieu des stades T/N. L'alignement embedding ↔ survie et la forêt sont
mutualisés (`src.survival_targets`, `src.survival_forest`).

```bash
python -m nnunet_survival.train           # réutilise le cache d'embeddings de tn + RandomSurvivalForest
```

---

## Repository structure

Le code est séparé en un paquet par tâche (`seg`, `tn`, `clinical_survival`,
`nnunet_survival`), plus `preprocessing/` (prépa des données, scripts autonomes). Les scripts
de soumission Slurm sont dans `slurm/`. `src/` regroupe le code partagé entre têtes (split,
embedding nnU-Net, métriques, alignement/forêt de survie).

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
│   ├── dataset.py               #   EmbeddingDataset (embedding nnU-Net ↔ cibles T/N)
│   ├── forest.py                #   RandomForestClassifier + GridSearchCV (balanced acc)
│   └── train.py                 #   point d'entrée  →  python -m tn.train
│
├── clinical_survival/           # Phase 2 — survie RFS (RandomSurvivalForest, données cliniques)
│   ├── dataset.py               #   ClinicalEncoder + ClinicalSurvivalDataset (variables cliniques)
│   └── train.py                 #   point d'entrée  →  python -m clinical_survival.train
│
├── nnunet_survival/             # Variante — survie RFS sur le bottleneck nnU-Net (même embedding que tn)
│   ├── dataset.py               #   pool_embedding (src) + alignement embedding ↔ cible de survie
│   └── train.py                 #   point d'entrée  →  python -m nnunet_survival.train
│
├── preprocessing/               # Prépa des données (scripts autonomes, hors pipeline d'entraînement)
│   ├── convert_nifti_Bq_SUV.py  #   conversion TEP BQML → SUVbw (bq_to_suv)
│   └── preprocess.py            #   resampling / recadrage / normalisation → python preprocessing/preprocess.py
│
├── src/                         # Code partagé entre plusieurs têtes
│   ├── split.py                 #   case_ids — lecture du split train/test du disque (toutes les têtes)
│   ├── nnunet_embedding.py      #   extraction du bottleneck nnU-Net + pool_embedding (tn, nnunet_survival)
│   ├── survival_targets.py      #   alignement embedding ↔ (RFS, événement) (nnunet_survival)
│   ├── metrics.py               #   balanced_accuracy, c_index
│   └── survival_forest.py       #   RandomSurvivalForest + GridSearchCV (clinical/nnunet_survival)
│
├── slurm/                       # scripts de soumission Slurm (train_seg, train_tn, …)
└── config.py                    # réglages nnU-Net, split, chemins directs vers results/
```
