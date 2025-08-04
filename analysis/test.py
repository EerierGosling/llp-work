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
from torchattacks import PGD

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


atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def pgd_attack(model, images, labels, targeted=True, eps=0.3, alpha=0.01, iters=40):
    """
    :param model: the model to attack
    :param images: original images
    :param labels: target labels
    :param targeted: if the attack is targeted
    :param eps: maximum perturbation
    :param alpha: step size
    :param iters: number of iterations
    :return: perturbed images
    Source: https://github.com/Harry24k/PGD-pytorch
    """
    if len(images.shape) != 4:
        raise ValueError("Input images must be 4-d tensors.")

    device = "cuda"

    if targeted:
        loss_fn = lambda out, lbl: -torch.nn.CrossEntropyLoss()(out, lbl)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)
    attack_images = images.clone().detach()
    attack_images.requires_grad = True

    for i in range(iters):
        outputs = model(attack_images)
        model.zero_grad()
        cost = loss_fn(outputs, labels).to(device)
        cost.backward()

        attack_images_grad = alpha * attack_images.grad.sign()
        attack_images = attack_images.detach() + attack_images_grad

        # Clip the attack_images to make sure they're valid images
        attack_images = torch.clamp(attack_images, min=0, max=1)
        attack_images = torch.clamp(attack_images, min=images-eps, max=images+eps).detach_()
        attack_images.requires_grad = True

    return attack_images

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


num_corect = [0]*5

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

    adv_inputs_fgsm = fgsm_attack(image, 0.24, data_grad)
    adv_input_fgsm = adv_inputs_fgsm.squeeze(0).to(device)

    adv_inputs_pgd = pgd_attack(model, image.unsqueeze(0), label.unsqueeze(0), eps=0.24)
    adv_input_pgd = adv_inputs_pgd.squeeze(0).to(device)

    adv_inputs_pgd_other = pgd_attack(model, image.unsqueeze(0), ((label+1)%len(class_names)).unsqueeze(0), eps=0.24)
    adv_input_pgd_other = adv_inputs_pgd_other.squeeze(0).to(device)

    adv_inputs_pgd_new = atk(image.unsqueeze(0), label.unsqueeze(0))

    inputs = [image, adv_input_fgsm, adv_input_pgd, adv_input_pgd_other, adv_inputs_pgd_new]

    for j, chosen_image in enumerate(inputs):
            
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
            num_corect[j] += 1



print(num_corect)