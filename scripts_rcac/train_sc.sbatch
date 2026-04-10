#!/bin/bash
#SBATCH --job-name=sc_pre_bc
#SBATCH --account=dinouye
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

# Ensure log directory exists
mkdir -p logs/slurm

# Activate environment
source ~/disk/anaconda3/etc/profile.d/conda.sh
conda activate catk

# Set master address for torchrun
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

cd /scratch/gautschi/chen5262/PythonProject/catk_starcraft
bash scripts_rcac/train_sc.sh
