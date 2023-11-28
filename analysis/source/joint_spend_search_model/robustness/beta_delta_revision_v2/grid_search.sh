#!/bin/bash

#SBATCH --job-name=rcc_2023_05_12d
#SBATCH --output=release/array_job/rcc_2023_05_12d.out
#SBATCH --error=release/array_job/rcc_2023_05_12d.err

#SBATCH --account=pi-ganong
#SBATCH --time=1-12:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user=jenders@rcc.uchicago.edu

#SBATCH --partition=caslake

#SBATCH --nodes=1

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

module load matlab
srun matlab -nodisplay -r < inf_horizon_het_results_search_cost_search_v3.m "jobname='$SLURM_JOB_NAME'; mainjobid=$SLURM_ARRAY_JOB_ID; taskid=$SLURM_ARRAY_TASK_ID;"