#!/bin/bash
#SBATCH --job-name=test_VTPAN
#SBATCH --output=logs/VTPAN/%x_%j.out
#SBATCH --error=logs/VTPAN/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kolomiez.andriy@gmail.com


# Activate your venv
echo "SLURM job started on $(hostname) at $(date)"
source /home/ak562fx/bac/venv/bin/activate

# Move into the script directory
cd /home/ak562fx/bac/grooming_detection

# Run the training script
python test_VTPAN.py

echo "Job complete at $(date)"