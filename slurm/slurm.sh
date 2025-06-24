#!/bin/sh
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=5     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:2          # the number of GPUs requested
#SBATCH --mem=100G             # memory 
#SBATCH -o slurm/outfile_%A_%a.txt      # send stdout to outfile
#SBATCH -e slurm/errfile_%A_%a.txt      # send stderr to errfile
#SBATCH -t 00:30:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=se0361@princeton.edu
#SBATCH --array=0           # creates 1 jobs with index 0

source ~/.bashrc
conda activate sofia
python training.py --job_id $SLURM_ARRAY_TASK_ID