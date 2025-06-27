#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kolomiez.andriy@gmail.com

echo "SLURM job started on $(hostname) at $(date)"
source /home/ak562fx/bac/venv/bin/activate
cd /home/ak562fx/bac/voice_detection

python scripts/test.py
echo "Job complete at $(date)"
