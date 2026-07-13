#!/bin/bash
#SBATCH --account=rrg-kjerbi
#SBATCH --job-name=build_h5
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=build_%j.out

# 1 job = 1 stade (passe STAGE et BRANCH en variables)
: "${STAGE:=S2}"
: "${BRANCH:=noica}"

source /home/alouis/mne_env/bin/activate 2>/dev/null || module load python
cd ~/scratch/cnn-dream-recall

python build_h5.py \
    --root /scratch/alouis/dream_bids \
    --deriv derivatives_250hz_dl \
    --branch "$BRANCH" \
    --stage "$STAGE" \
    --out "data/all_${STAGE}_${BRANCH}.h5" \
    --win 1250
