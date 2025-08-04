import torchvision.models as models
import torch.nn as nn
import torch
import glob
from types import SimpleNamespace
from generate_data import *
import os
import torch
import torchvision.datasets as datasets
from torchvision import transforms
import random
import numpy as np
import pandas as pd
import uuid
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import matplotlib.pyplot as plt
from generate_data import *
from analysis_functions import *
import glob

print("starting")
id = uuid.uuid4().hex[:8]
# region - setup
epsilon_options = list(np.arange(0, 0.6, 0.02))
timestep_options = [100]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# image setup

dataset_transformed = datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)



model = models.resnet34()
model.maxpool = nn.Identity()
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

model_folder = f'/n/fs/visualai-scr/temp_LLP/sofia/llp-work/trained-models/adversarial/resnet34/0.24'
files = glob.glob(os.path.join(model_folder, "*.pth"))

most_recent = max(files, key=os.path.getmtime)

# Move model to device BEFORE loading weights to ensure correct dtype
model = model.to(device)
model.load_state_dict(torch.load(most_recent, map_location=device))


# https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

num_correct_eval = 0
num_correct_train = 0

num_correct_eval_adv = 0
num_correct_train_adv = 0

for i in range(100):
    random_index = random.randint(0, len(dataset_transformed) - 1)
    image, label = dataset_transformed[random_index]
    image = image.to(device)
    label = torch.tensor(label).to(device)
    image.requires_grad = True

    # Forward pass
    output = model(image.unsqueeze(0))
    loss = torch.nn.functional.cross_entropy(output, label.unsqueeze(0))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data

    adv_inputs = fgsm_attack(image, 0.24, data_grad)
    adv_input = adv_inputs.squeeze(0).to(device)

    for image_option in range(2):
        chosen_image = adv_input if image_option == 0 else image
        for j in range(2):
            if j == 0:
                model.eval()
            else:
                model.train()
            
            model = model.to(device)

            input_batch = chosen_image.unsqueeze(0)
            input_batch = input_batch.to(device)
            # Remove: input_batch.requires_grad = True
            # Only set requires_grad on leaf tensors, i.e., chosen_image if needed
            if not chosen_image.requires_grad:
                chosen_image.requires_grad = True

            output = model(input_batch)

            _, predicted_idx = torch.max(output, 1)
            predicted_class = predicted_idx.item()

            score = output[0, label]

            if predicted_class == label:
                if j == 0:
                    if image_option == 0:
                        num_correct_eval_adv += 1
                    else:
                        num_correct_eval += 1

                else:
                    if image_option == 0:
                        num_correct_train_adv += 1
                    else:
                        num_correct_train += 1



print(num_correct_eval, num_correct_train)
print(num_correct_eval_adv, num_correct_train_adv)