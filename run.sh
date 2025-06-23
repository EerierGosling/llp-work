#!/bin/sh
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:1          # the number of GPUs requested
#SBATCH --mem=50G             # memory 
#SBATCH -o outfile            # send stdout to outfile
#SBATCH -e errfile            # send stderr to errfile
#SBATCH -t 00:01:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=<your-netid>@princeton.edu

conda activate sofia
python main.py