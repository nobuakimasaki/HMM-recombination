#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=masakin@uw.edu
#SBATCH --job-name=recomb_smk_single
#SBATCH --output=logs/%x_%j.out    # %x = job name, %j = job ID
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1-00:00:00          # 1 day max runtime
#SBATCH --cpus-per-task=30
#SBATCH --mem=30G

# 1. load the Miniforge shell hook
source /app/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh

# 2. activate the environment
conda activate recomb-snake

# 3. run Snakemake
snakemake --cores "$SLURM_CPUS_PER_TASK" \
          --nolock \
          --printshellcmds --rerun-incomplete