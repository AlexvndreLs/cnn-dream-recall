#!/bin/bash
#SBATCH --job-name=cnn_smoke
#SBATCH --time=00:20:00
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=smoke_%j.out

# --- environnement (a adapter a ton setup fir) ---
module load python/3.11 cuda 2>/dev/null
source ~/scratch/venvs/cnn/bin/activate 2>/dev/null || {
    python -m venv ~/scratch/venvs/cnn
    source ~/scratch/venvs/cnn/bin/activate
    pip install --no-index torch scikit-learn h5py numpy 2>/dev/null \
        || pip install torch scikit-learn h5py numpy
}

cd ~/scratch/cnn-dream-recall
python smoke_test.py
