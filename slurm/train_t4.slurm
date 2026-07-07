#!/bin/bash
#SBATCH --job-name=train_t4      # nom du job
#SBATCH --output=train_t4.out   # fichier de sortie
#SBATCH --error=train_t4.err    # fichier d'erreur
#SBATCH --constraint=h100               # demander des GPU H100 80 Go
#SBATCH --nodes=1                       # reserver 1 noeud
#SBATCH --ntasks=1                      # 1 tache shell
#SBATCH --gres=gpu:1                    # reserver 1 GPU
#SBATCH --cpus-per-task=8               # reserver 8 CPU
#SBATCH --time=10:00:00                 # temps maximal d'allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread            # desactiver l'hyperthreading
#SBATCH --account=ehe@h100              # comptabilite H100

module purge
conda deactivate
module load arch/h100
module load pytorch-gpu/py3/2.4.0

source /lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/hecktor_venv/bin/activate
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/hecktor_venv/lib/python3.11/site-packages:$PYTHONPATH

export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Remove system cuDNN from LD_LIBRARY_PATH so PyTorch uses its bundled version
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'cudnn' | tr '\n' ':' | sed 's/:$//')

# Disable torch.compile (Triton JIT fails on this cluster due to missing libcuda.so.1 in gcc -L paths)
export TORCH_COMPILE_DISABLE=1

set -x

cd /lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal
python -m TN.t4
