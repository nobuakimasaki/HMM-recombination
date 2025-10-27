#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL          # when to notify: BEGIN|END|FAIL|ALL|REQUEUE…
#SBATCH --mail-user=masakin@uw.edu   # where to send it
#SBATCH --job-name=recomb_smk
#SBATCH --output=logs/%x_%A_%a.out       # logs/recomb_smk_JOBID_TASKID.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=3-12:00:00                  # wall-clock limit
#SBATCH --cpus-per-task=20                # Snakemake will use these cores (use 36 for efficiency)
#SBATCH --mem=30G
#SBATCH --array=0-39%13                     # updated automatically by submit_all.sh

# 1. load the Miniforge shell hook
source /app/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh

# 2. activate the all-in-one environment
conda activate recomb-snake

# 3. pick the right config for this array index
CONFIGS=($(ls configs/config_*.yaml | grep -v 'config_small.yaml'))
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Running Snakemake for $CFG on $SLURM_CPUS_PER_TASK cores"

# 4. run the workflow (single–env mode)
snakemake --cores "$SLURM_CPUS_PER_TASK" \
          --configfile "$CFG" \
          --nolock \
          --printshellcmds --rerun-incomplete

