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
from sofias_generate_data import *
from analysis_functions import *
import os

print("starting")

# region - setup
epsilon_options = np.arange(0, 0.3, 0.02)
timestep_options = np.arange(0, 1000, 100)

website = False

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


if website:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()
    image_no_transform = Image.open(f"website-images/{args.file_name}").convert('RGB')
    image_no_transform = image_no_transform.resize((32, 32))
    input_tensor = transform(image_no_transform)
    image = input_tensor
    label = 0
else:
    random_idx = random.randint(0, len(dataset_transformed) - 1)
    image, label = dataset_transformed[random_idx]
    image_no_transform, _ = dataset[random_idx]

input_batch = image.unsqueeze(0)
input_batch = input_batch.to(device)
input_batch.requires_grad = True
# endregion

print("setup done")

classifier_gradients = []

for epsilon in epsilon_options:
    gradients, _, _ = run_classifer(input_batch, adversarial=True, epsilon=epsilon, device=device)

    classifier_gradients.append({ "epsilon": epsilon, "gradients": gradients })
    print(f"epsilon {epsilon} done")


diffusion_gradients = []

for timestep in timestep_options:
    gradients, saliency_map_diffusion = run_diffusion(image, label, timestep, device=device)

    diffusion_gradients.append({ "timestep": timestep, "gradients": gradients })

    print(f"timestep {timestep} done")


results = []

for classifier_gradient in classifier_gradients:
    for diffusion_gradient in diffusion_gradients:
        mse = mean_squared_error(classifier_gradient["gradients"], diffusion_gradient["gradients"])
        cosine_sim = cosine_similarity(classifier_gradient["gradients"], diffusion_gradient["gradients"])

        results.append({
            "epsilon": classifier_gradient["epsilon"],
            "timestep": diffusion_gradient["timestep"],
            "mse": mse,
            "cosine_similarity": cosine_sim
        })
        print(f"epsilon {classifier_gradient['epsilon']}, timestep {diffusion_gradient['timestep']} done")

analysis_id = str(uuid.uuid4())[:8]
csv_filename = f"/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis-data/{analysis_id}.csv"

# Create directory if it doesn't exist
os.makedirs("/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis-data", exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(csv_filename, index=False)

# region - show images
if False:
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(np.array(image_no_transform).transpose(1, 2, 0))
    plt.title(f"Original: {class_names[label]}")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(saliency_maps[0], cmap='viridis')
    plt.title(f"Predicted: {class_names[predicted_classes[0]]}")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(saliency_maps[1], cmap='viridis')
    plt.title(f"Predicted: {class_names[predicted_classes[1]]}")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(np.mean(saliency_map_diffusion, axis=0), cmap='viridis')
    plt.title(f"Diffusion Gradient\nClass: {class_names[label]}")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/website.png")
#endregion