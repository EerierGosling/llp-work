import sys
import os

learning_rate = 0.001
weight_decay = 0.01
epsilon = [0.01, 0.1, 0.2]
adversarial_ratio = [0.1, 0.2, 0.3]

for i in range(9):
    os.system(f"sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 01:00:00 --wrap=\"source ~/.bashrc && conda activate sofia && python adversarial.py --learning_rate {learning_rate} --weight_decay {weight_decay} --epsilon {epsilon[i % 3]} --adversarial_ratio {adversarial_ratio[i // 3]}\"")