#!/bin/bash

#SBATCH --job-name=append_2023_05_12d
#SBATCH --output=release/array_job/append_2023_05_12d.out
#SBATCH --error=release/array_job/append_2023_05_12d.err

#SBATCH --account=pi-ganong
#SBATCH --time=01:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user=jenders@rcc.uchicago.edu

#SBATCH --partition=caslake

#SBATCH --nodes=1

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

module load matlab
srun matlab -nodisplay -r < append_results_grid_search_v3.m "jobname='$SLURM_JOB_NAME'; mainjobid=$SLURM_ARRAY_JOB_ID; taskid=$SLURM_ARRAY_TASK_ID;"