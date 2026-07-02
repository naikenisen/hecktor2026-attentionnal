#!/usr/bin/env bash
# Copie les sous-dossiers d'un dossier source vers une destination, en ne gardant
# que les fichiers dont le nom NE contient PAS "CT" ni "PT" (donc les masques).
set -euo pipefail

SRC="/beegfs/data/work/imvia/in156281/datasets/hecktor_dataset_preprocessed_suv"                 # dossier source (un sous-dossier par patient)
DST="/beegfs/data/work/imvia/in156281/datasets/dataset_masks"           # dossier destination

for subdir in "$SRC"/*/; do
    name="$(basename "$subdir")"
    mkdir -p "$DST/$name"
    for file in "$subdir"*; do
        base="$(basename "$file")"
        if [[ "$base" != *CT* && "$base" != *PT* ]]; then
            cp "$file" "$DST/$name/"
        fi
    done
    echo "$name → $DST/$name"
done
