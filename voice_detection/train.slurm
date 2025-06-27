#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kolomiez.andriy@gmail.com

echo "SLURM job started on $(hostname) at $(date)"
source /home/ak562fx/bac/venv/bin/activate
cd /home/ak562fx/bac/voice_detection

python scripts/train.py
echo "Job complete at $(date)"
