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
import os

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

transform_no_normalize = transforms.Compose([
    transforms.ToTensor(),
])

# image setup

dataset_transformed = datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)

dataset = datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True,
    transform=transform_no_normalize
)

print("setup done")

# endregion

# region - website stuff
# parser = argparse.ArgumentParser()
# parser.add_argument('--file_name', type=str)
# args = parser.parse_args()
# image_no_transform = Image.open(f"website-images/{args.file_name}").convert('RGB')
# image_no_transform = image_no_transform.resize((32, 32))
# input_tensor = transform(image_no_transform)
# image = input_tensor
# label = 0
# endregion

all_data = []

random_index = random.randint(0, len(dataset_transformed) - 1)
image, label = dataset_transformed[random_index]
image_no_transform, _ = dataset[random_index]
print("images selected")


output_dir = f"/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/analysis-images/gradients/run8"
os.makedirs(output_dir, exist_ok=True)
image_paths = []

for epsilon in epsilon_options:

    gradients_classifier, _, _ = run_classifer(image, label, adversarial=True, epsilon=epsilon, device=device)
    classifier_gradients = [{ "epsilon": epsilon, "gradients": gradients_classifier }]
    print(f"epsilon {epsilon} done")
    print("done with classifier")

    # Loop over timesteps
    for timestep in timestep_options:
        print(f"Processing timestep {timestep}")
        gradients_diffusion, _ = run_diffusion(image, label, timestep, device=device)
        diffusion_gradients = [{ "timestep": timestep, "gradients": gradients_diffusion }]

        results = []
        for classifier_gradient in classifier_gradients:
            for diffusion_gradient in diffusion_gradients:
                mse = mean_squared_error(classifier_gradient["gradients"], diffusion_gradient["gradients"], device=device)
                cosine_sim = cosine_similarity(classifier_gradient["gradients"], diffusion_gradient["gradients"])
                results.append({
                    "epsilon": classifier_gradient["epsilon"],
                    "timestep": diffusion_gradient["timestep"],
                    "mse": mse,
                    "cosine_similarity": cosine_sim
                })

        # region - show images
        plt.figure(figsize=(20, 5))

        # Show original image
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(image_no_transform).transpose(1, 2, 0))
        plt.title(f"Original: {class_names[label]}")
        plt.axis('off')

        # Show classifier gradients (mean across channels)
        plt.subplot(1, 3, 2)
        classifier_grad = classifier_gradients[0]["gradients"]
        if hasattr(classifier_grad, "detach"):
            classifier_grad = classifier_grad.detach().cpu().numpy()
        if classifier_grad.ndim == 4 and classifier_grad.shape[0] == 1:
            classifier_grad = np.squeeze(classifier_grad, axis=0)  # shape (3, 32, 32)
        if classifier_grad.ndim == 3 and classifier_grad.shape[0] == 3:
            grad_img = classifier_grad.mean(axis=0)  # shape (32, 32)
        else:
            grad_img = classifier_grad  # fallback
        plt.imshow(grad_img, cmap='viridis')
        plt.title(f"Classifier Gradients\nmse: {results[0]['mse']:.4f}\ncosine similarity: {results[0]['cosine_similarity']:.4f} mean:{classifier_grad.mean():.4f}")
        plt.axis('off')

        # Show diffusion gradients (mean across channels)
        plt.subplot(1, 3, 3)
        diffusion_grad = diffusion_gradients[0]["gradients"]
        if hasattr(diffusion_grad, "detach"):
            diffusion_grad = diffusion_grad.detach().cpu().numpy()
        if diffusion_grad.ndim == 4 and diffusion_grad.shape[0] == 1:
            diffusion_grad = np.squeeze(diffusion_grad, axis=0)
        if diffusion_grad.ndim == 3 and diffusion_grad.shape[0] == 3:
            grad_img = diffusion_grad.mean(axis=0)
        else:
            grad_img = diffusion_grad
        plt.imshow(grad_img, cmap='viridis')
        plt.title(f"Diffusion Gradients\nmse: {results[0]['mse']:.4f}\ncosine similarity: {results[0]['cosine_similarity']:.4f} mean:{diffusion_grad.mean():.4f}")
        plt.axis('off')

        plt.tight_layout()
        out_path = f"{output_dir}/{epsilon}_{id}.png"
        plt.savefig(out_path)
        plt.close()
        image_paths.append(out_path)
# endregion


from PIL import Image

if image_paths:
    images = [Image.open(p) for p in image_paths]
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    total_height = sum(heights)
    stacked_img = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for img in images:
        stacked_img.paste(img, (0, y_offset))
        y_offset += img.size[1]
    stacked_img.save(f"{output_dir}/stacked_{id}.png")