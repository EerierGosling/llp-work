import sys
import os

for i in range(9):
    os.system(f"sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 01:00:00 --wrap=\"source ~/.bashrc && conda activate sofia && python analysis.py\"")