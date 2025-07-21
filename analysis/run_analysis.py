import sys
import os

os.system(f"sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 01:00:00 --output=/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/job_%j.out --error=/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/job_%j.err --chdir=/n/fs/visualai-scr/temp_LLP/sofia/llp-work --wrap=\"source ~/.bashrc && conda activate sofia && python /n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis.py\"")

# sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 01:00:00 --wrap="source ~/.bashrc && conda activate sofia && python /n/fs/visualai-scr/temp_LLP/sofia/llp-work/minimal-diffusion/analysis.py"