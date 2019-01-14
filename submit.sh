#!/bin/sh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
srun $@
