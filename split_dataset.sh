#!/usr/bin/env bash
#
# Répartit les sous-dossiers patients de hecktor_dataset_suv en train/ et test/
# selon la colonne "split" de HECKTOR_2026_training_data.csv.
#   split == train -> train/
#   split == test  -> test/
#
# Idempotent et re-jouable : un patient est cherché à plat sous le dataset OU déjà dans
# train/ ou test/ ; il n'est déplacé que s'il n'est pas déjà dans le bon dossier. On peut
# donc relancer ce script après avoir modifié la colonne split pour resynchroniser les
# dossiers.
#
set -euo pipefail

CSV="tables/HECKTOR_2026_training_data.csv"
DATASET="/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/dataset/hecktor_dataset_suv"

# Résolution du CSV relatif à la racine du dépôt (répertoire du script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$CSV" ]] || CSV="$SCRIPT_DIR/$CSV"

if [[ ! -f "$CSV" ]]; then
    echo "Erreur : CSV introuvable : $CSV" >&2
    exit 1
fi
if [[ ! -d "$DATASET" ]]; then
    echo "Erreur : dataset introuvable : $DATASET" >&2
    exit 1
fi

mkdir -p "$DATASET/train" "$DATASET/test"

# Indices des colonnes PatientID et split (en cas de réordonnancement du CSV)
header=$(head -1 "$CSV")
IFS=',' read -ra cols <<< "$header"
pid_idx=-1; split_idx=-1
for i in "${!cols[@]}"; do
    case "${cols[$i]}" in
        PatientID) pid_idx=$i ;;
        split)     split_idx=$i ;;
    esac
done
if [[ $pid_idx -lt 0 || $split_idx -lt 0 ]]; then
    echo "Erreur : colonnes PatientID/split introuvables dans l'en-tête" >&2
    exit 1
fi

moved=0; missing=0; skipped=0
# tail -n +2 : on saute l'en-tête
while IFS=',' read -ra fields; do
    pid="${fields[$pid_idx]}"
    split="${fields[$split_idx]}"
    [[ -z "$pid" ]] && continue

    case "$split" in
        train) dest="train" ;;
        test)  dest="test" ;;
        *)     echo "  ! split inconnu '$split' pour $pid (ignoré)"; skipped=$((skipped+1)); continue ;;
    esac

    # Déjà dans le bon dossier → rien à faire (relance idempotente).
    if [[ -d "$DATASET/$dest/$pid" ]]; then
        continue
    fi

    # Localise le patient : à plat sous le dataset, ou dans l'autre split.
    other=$([[ "$dest" == "train" ]] && echo test || echo train)
    if [[ -d "$DATASET/$pid" ]]; then
        src="$DATASET/$pid"
    elif [[ -d "$DATASET/$other/$pid" ]]; then
        src="$DATASET/$other/$pid"
    else
        echo "  ! dossier manquant : $pid"
        missing=$((missing+1))
        continue
    fi

    mv "$src" "$DATASET/$dest/"
    moved=$((moved+1))
done < <(tail -n +2 "$CSV")

echo "Terminé : $moved déplacés, $missing manquants, $skipped ignorés."
