#!/bin/bash
#SBATCH --job-name=sc_train
#SBATCH --account=dinouye
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

# Usage: sbatch scripts_rcac/run.sbatch train_smart.sh

SCRIPT=${1:?Usage: sbatch scripts_rcac/run.sbatch <script_name.sh>}

mkdir -p logs/slurm

source ~/disk/anaconda3/etc/profile.d/conda.sh
conda activate catk

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

cd /scratch/gautschi/chen5262/PythonProject/catk_starcraft
bash scripts_rcac/$SCRIPT
