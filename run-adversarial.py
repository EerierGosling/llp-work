import sys
import os

learning_rate = [0.005, 0.001, 0.0005]
weight_decay = 0.01
epsilon = [0.001, 0.01, 0.02]
adversarial_ratio = 0.2

# for i in range(9):
#     os.system(f"sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 05:00:00 --wrap=\"source ~/.bashrc && conda activate sofia && python adversarial.py --learning_rate {learning_rate[i // 3]} --weight_decay {weight_decay} --epsilon {epsilon[i % 3]} --adversarial_ratio {adversarial_ratio} --adversarial_training True\"")

for i in range(9):
    os.system(f"sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 05:00:00 --wrap=\"source ~/.bashrc && conda activate sofia && python adversarial.py --learning_rate {learning_rate[i // 3]} --weight_decay {weight_decay} --epsilon {epsilon[i % 3]} --adversarial_ratio {adversarial_ratio} --adversarial_training False\"")