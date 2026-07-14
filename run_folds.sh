#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --job-name=cnn_fold
#SBATCH --array=0-17
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=fc30555
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=%x_%A_%a.out

# 1 job = 1 fold. Passe STAGE (et FAITHFUL=1 pour le mode fidele).
: "${STAGE:=S2}"
: "${BRANCH:=noica}"
: "${OUTDIR:=res_${STAGE}}"

FAITH=""
[ "${FAITHFUL:-0}" = "1" ] && FAITH="--faithful" && OUTDIR="${OUTDIR}_faithful"

source /home/alouis/mne_env/bin/activate
cd ~/scratch/cnn-dream-recall
mkdir -p "$OUTDIR"

python run.py \
    --h5 "/scratch/alouis/cnn_data/all_${STAGE}_${BRANCH}.h5" \
    --device cuda --workers 4 \
    --fold "$SLURM_ARRAY_TASK_ID" \
    $FAITH \
    --out "${OUTDIR}/fold_$(printf '%02d' $SLURM_ARRAY_TASK_ID).npz"
