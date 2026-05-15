#!/bin/ksh

# Modify these variables to adapt the sh script to your ccub accout and workgroup
USER_NAME=am769644
GROUP_NAME=c-2iia

# modify theses variables according to your project directory name and virtual environment directory name
PROJECT_NAME=transformer-unet
VENV_SUBFOLDER=transformer-unet-old
PYTHON_SITE_PACKAGES_VERSION=3.9

# This part should not be modified
BASE_BEEGFS=/beegfs/data/work/$GROUP_NAME/$USER_NAME
BASE_WORK=/work/$GROUP_NAME/$USER_NAME
PROJECT_DIR=$BASE_BEEGFS/$PROJECT_NAME
VENV_DIR=$BASE_BEEGFS/$VENV_SUBFOLDER/venv

cd "$WORKDIR"
cd "$PROJECT_DIR"

module load python
mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip3 install --prefix="$VENV_DIR" -r requirements.txt
export PYTHONPATH="$VENV_DIR/lib/python$PYTHON_SITE_PACKAGES_VERSION/site-packages:$PYTHONPATH"
pip3 list
