#!/bin/ksh
#$ -q gpu
#$ -o retrain_seg.out
#$ -j y
#$ -N retrain_seg

set -e

USER_NAME=in156281
GROUP_NAME=imvia
PROJECT_NAME=hecktor2026-attentionnal
VENV_NAME=hecktor_venv

BASE_BEEGFS=/beegfs/data/work/$GROUP_NAME/$USER_NAME
PROJECT_DIR=$BASE_BEEGFS/$PROJECT_NAME
VENV_DIR=$BASE_BEEGFS/venvs/$VENV_NAME
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"
module load python
export PYTHONPATH="$VENV_DIR/lib/python3.9/site-packages:$PYTHONPATH"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python retrain_seg.py
