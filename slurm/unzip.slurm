#!/bin/bash
#SBATCH --job-name=unzip_dataset        # Nom du job
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # Nombre total de processus MPI
#SBATCH --hint=nomultithread       # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=00:20:00            # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=unzip.out   # Nom du travail
#SBATCH --error=unzip.out    # Nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --qos=qos_cpu-dev
  

module purge
conda deactivate
module load petsc/3.20.1-mpi
  
view_args="-log_view" # to avoid slowing down due to -log_view_gpu_time
  
cpu_args="-dm_mat_type mpiaij -dm_vec_type mpi"
  
set -x
unzip /lustre/fsn1/projects/rech/ehe/udq27fb/hecktor2026-attentionnal/dataset-hecktor/dataset.zip