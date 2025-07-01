import sys
import os

learning_rate = [0.005, 0.001, 0.0005]
weight_decay = 0.01

for i in range(3):
    os.system(f"sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 05:00:00 --wrap=\"source ~/.bashrc && conda activate sofia && python resnet-classifier.py --learning_rate {learning_rate[i]} --weight_decay {weight_decay}\"")