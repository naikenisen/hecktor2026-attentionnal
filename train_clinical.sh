#!/bin/ksh 
#$ -q gpu
#$ -o train_clinical.out
#$ -j y
#$ -N train

set -e

# Modify these variables to adapt the sh script to your ccub accout and workgroup
USER_NAME=in156281
GROUP_NAME=imvia

# modify theses variables according to your project directory name and virtual environment directory name
PROJECT_NAME=hecktor2026-attentionnal
VENV_NAME=hecktor_venv

# this part should not be modified
BASE_BEEGFS=/beegfs/data/work/$GROUP_NAME/$USER_NAME
BASE_WORK=/work/$GROUP_NAME/$USER_NAME
PROJECT_DIR=$BASE_BEEGFS/$PROJECT_NAME
VENV_DIR=$BASE_BEEGFS/venvs/$VENV_NAME
cd "$WORKDIR"
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"
module load python
export PYTHONPATH="$VENV_DIR/lib/python3.9/site-packages:$PYTHONPATH"

python train_clinical.py