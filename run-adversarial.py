import sys
import os

learning_rate = 0.0005
weight_decay = 0.01
epsilon = 0.01
adversarial_ratio = 0.2

for i in range(1):
    os.system(f"sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 05:00:00 --wrap=\"source ~/.bashrc && conda activate sofia && python adversarial.py --learning_rate {learning_rate} --weight_decay {weight_decay} --epsilon {epsilon} --adversarial_ratio {adversarial_ratio} --adversarial_training\"")

for i in range(1):
    os.system(f"sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 05:00:00 --wrap=\"source ~/.bashrc && conda activate sofia && python adversarial.py --learning_rate {learning_rate} --weight_decay {weight_decay} --epsilon {epsilon} --adversarial_ratio {adversarial_ratio}\"")