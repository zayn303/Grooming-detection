#!/bin/bash
#SBATCH --job-name=groom-api
#SBATCH --output=logs/api_%j.out
#SBATCH --error=logs/api_%j.err
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=40-00:00:00
#SBATCH --requeue


echo "SLURM job started on $(hostname) at $(date)"
source /home/ak562fx/bac/venv/bin/activate

cd /home/ak562fx/bac  

export PYTHONPATH=$(pwd)  
export PYTHONUNBUFFERED=1

cd API    


# Suppress known unwanted warnings/noise
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore::UserWarning"

# Start the FastAPI server on port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info

echo "Job complete at $(date)"