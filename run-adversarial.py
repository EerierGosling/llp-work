import subprocess
import sys
import os

learning_rate = 0.001
weight_decay = 0.01
epsilon = [0.01, 0.1, 0.2]
adversarial_ratio = [0.1, 0.2, 0.3]

for i in range(9):
    command = f"python adversarial.py --learning_rate {learning_rate} --weight_decay {weight_decay} --epsilon {epsilon[i % 3]} --adversarial_ratio {adversarial_ratio[i // 3]}"
    print(f"Running command: {command}")

    os.system(command)