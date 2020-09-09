#!/bin/bash

#SBATCH --job-name=RL-Praktikum
#SBATCH --output=res.txt
#SBATCH --partition=Gobi
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=3000

srun hostname
srun python3 train_main.py
 
