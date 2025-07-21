import sys
import os
import numpy as np

learning_rate = 0.0005
weight_decay = 0.01
epsilon_range = np.arange(0, 0.3, 0.02)
adversarial_ratio = 0.2

common_test_epsilons = ""
for i in np.arange(0, 0.3, 0.08):
    common_test_epsilons += f"{i},"

common_test_epsilons = common_test_epsilons.strip(',')

print(f"common_test_epsilons: {common_test_epsilons}")

def run_job(flags):
    os.system(f"sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 05:00:00 --output=/n/fs/visualai-scr/temp_LLP/sofia/llp-work/logs/job_%j.out --error=/n/fs/visualai-scr/temp_LLP/sofia/llp-work/logs/job_%j.err --chdir=/n/fs/visualai-scr/temp_LLP/sofia/llp-work --wrap=\"source ~/.bashrc && conda activate sofia && python adversarial.py {flags}\"")

for epsilon in epsilon_range:
    run_job(f"--learning_rate {learning_rate} --weight_decay {weight_decay} --epsilon {epsilon} --common_test_epsilons {common_test_epsilons} --adversarial_ratio {adversarial_ratio} --adversarial_training")
    print(f"ran {epsilon}")


run_job(f"--learning_rate {learning_rate} --weight_decay {weight_decay} --epsilon 0 --common_test_epsilons {common_test_epsilons} --adversarial_ratio {adversarial_ratio}")