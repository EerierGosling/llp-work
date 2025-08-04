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

# region - setup
epsilon_options = np.arange(0, 0.6, 0.02)
timestep_options = np.arange(0, 1000, 50)

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

num_images = 30

for i in range(num_images):
    random_index = random.randint(0, len(dataset_transformed) - 1)
    image, label = dataset_transformed[random_index]
    image_no_transform, _ = dataset[random_index]
    print("images selected")

    classifier_gradients = []

    for epsilon in epsilon_options:
        gradients, _, _ = run_classifer(image, label, adversarial=True, epsilon=epsilon, device=device)

        classifier_gradients.append({ "epsilon": epsilon, "gradients": gradients })
        print(f"epsilon {epsilon} done")
    print("done with classifier")


    diffusion_gradients = []

    for timestep in timestep_options:
        gradients, saliency_map_diffusion = run_diffusion(image_no_transform, label, timestep, device=device)

        diffusion_gradients.append({ "timestep": timestep, "gradients": gradients })
        print(f"timestep {timestep} done")

    print("done with diffusion")


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
            # print(f"epsilon {classifier_gradient['epsilon']}, timestep {diffusion_gradient['timestep']} done")

    print("done getting diff")

    analysis_id = str(uuid.uuid4())[:8]
    csv_filename = f"/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/data/{analysis_id}.csv"

    # Create directory if it doesn't exist
    os.makedirs("/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/data", exist_ok=True)

    df = pd.DataFrame(results)
    all_data.append(df)
    df.to_csv(csv_filename, index=False)

    print(f"iteration {i+1} done")

average_df = pd.concat(all_data).groupby(['epsilon', 'timestep']).mean().reset_index()
average_csv_filename = f"/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/data/average/{num_images}_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
average_df.to_csv(average_csv_filename, index=False)

# region - show images
# plt.figure(figsize=(20, 5))

# plt.subplot(1, 4, 1)
# plt.imshow(np.array(image_no_transform).transpose(1, 2, 0))
# plt.title(f"Original: {class_names[label]}")
# plt.axis('off')

# plt.subplot(1, 4, 2)
# plt.imshow(saliency_maps[0], cmap='viridis')
# plt.title(f"Predicted: {class_names[predicted_classes[0]]}")
# plt.axis('off')

# plt.subplot(1, 4, 3)
# plt.imshow(saliency_maps[1], cmap='viridis')
# plt.title(f"Predicted: {class_names[predicted_classes[1]]}")
# plt.axis('off')

# plt.subplot(1, 4, 4)
# plt.imshow(np.mean(saliency_map_diffusion, axis=0), cmap='viridis')
# plt.title(f"Diffusion Gradient\nClass: {class_names[label]}")
# plt.axis('off')

# plt.tight_layout()
# plt.savefig(f"/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis-images/website.png")
#endregion