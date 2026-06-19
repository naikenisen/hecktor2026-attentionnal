#!/bin/bash

ZIP="/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/dataset-hecktor/dataset.zip"
DEST="/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/dataset-hecktor/dataset_hecktor"

echo "📦 Extraction de : $ZIP"
echo "📁 Destination   : $DEST"
echo ""

mkdir -p "$DEST"

# Nombre total de fichiers dans le zip
TOTAL=$(unzip -l "$ZIP" | tail -1 | awk '{print $2}')

echo "📊 Fichiers à extraire : $TOTAL"
echo ""

# Extraction avec comptage en temps réel
TRAITES=0

unzip -o "$ZIP" -d "$DEST" | while IFS= read -r ligne; do
  # unzip affiche "inflating: ..." ou "extracting: ..." pour chaque fichier
  if echo "$ligne" | grep -qE "^\s+(inflating|extracting|creating):"; then
    TRAITES=$((TRAITES + 1))
    POURCENTAGE=$(awk "BEGIN { printf \"%.1f\", ($TRAITES/$TOTAL)*100 }")

    # Barre de progression
    BAR_WIDTH=40
    FILLED=$(awk "BEGIN { printf \"%d\", ($TRAITES/$TOTAL)*$BAR_WIDTH }")
    EMPTY=$((BAR_WIDTH - FILLED))
    BAR=$(printf "%${FILLED}s" | tr ' ' '█')$(printf "%${EMPTY}s" | tr ' ' '░')

    # Affichage sur la même ligne (écrasement)
    printf "\r  [%s] %s%%  %d / %d" "$BAR" "$POURCENTAGE" "$TRAITES" "$TOTAL"
  fi
done

echo ""
echo ""
echo "✅ Extraction terminée !"
echo "📁 Contenu de $DEST :"
ls -lh "$DEST"