#!/bin/ksh 
#$ -q gpu
#$ -o train.out
#$ -j y
#$ -N train

set -e

# Modify these variables to adapt the sh script to your ccub accout and workgroup
#am769644 #in156281
#c-2iia   #imvia
USER_NAME=in156281
GROUP_NAME=imvia

# modify theses variables according to your project directory name and virtual environment directory name
PROJECT_NAME=transformer-unet
VENV_SUBFOLDER=transformer-unet-old

# Set your Wandb API key here
WANDB_API_KEY="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"

# this part should not be modified
BASE_BEEGFS=/beegfs/data/work/$GROUP_NAME/$USER_NAME
BASE_WORK=/work/$GROUP_NAME/$USER_NAME
PROJECT_DIR=$BASE_BEEGFS/$PROJECT_NAME
VENV_DIR=$BASE_BEEGFS/$VENV_SUBFOLDER/venv
cd "$WORKDIR"
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"
module load python
export PYTHONPATH="$VENV_DIR/lib/python3.9/site-packages:$PYTHONPATH"
export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_CACHE_DIR="$BASE_WORK/.cache/wandb"
export WANDB_CONFIG_DIR="$BASE_WORK/.config/wandb"
export WANDB_DATA_DIR="$BASE_WORK/.local/share/wandb"
python train.py