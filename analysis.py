import torch
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision import transforms
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import datetime
import argparse
from types import SimpleNamespace
import matplotlib.pyplot as plt
from sofias_generate_data import *

# setup

parser = argparse.ArgumentParser()

adversarial = "2025-07-03 14:34:41"
non_adversarial = "2025-07-03 14:34:43"

website = False

parser.add_argument('--file_name', type=str)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# args = parser.parse_args()
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform2 = transforms.Compose([
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
    transform=transform2
)


if website:
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



# classifer

saliency_maps = []
predicted_classes = []

for i in range(2):
    name = [adversarial, non_adversarial][i]

    model = models.resnet34()
    model.maxpool = nn.Identity()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    PATH = f'./trained-models/{name}.pth'
    model.load_state_dict(torch.load(PATH))

    model.eval()
    model = model.to(device)

    output = model(input_batch)

    _, predicted_idx = torch.max(output, 1)
    predicted_classes.append(predicted_idx.item())

    print("predicted:" + class_names[label])
    print("actual:" + class_names[predicted_classes[i]])

    score = output[0, predicted_classes[i]]

    gradients = torch.autograd.grad(outputs=score, inputs=input_batch)[0]

    saliency = torch.abs(gradients)

    saliency_maps.append(torch.max(saliency, dim=1)[0].squeeze().cpu().numpy())

    saliency_maps[i] = (saliency_maps[i] - saliency_maps[i].min()) / (saliency_maps[i].max() - saliency_maps[i].min() + 1e-8)




# diffusion model


# Specify the configuration
args =  SimpleNamespace(
    dataset='cifar10',
    timesteps=1000,
    device='cuda',
    batch_size=8, 
    guidance_scale=2.0,
    ddim=True,
    sampling_steps=50,
    pretrained_ckpt='/n/fs/wy-project/minimal-diffusion/trained_models3/UNet_cifar10-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt',
    arch='UNet',
    diffusion_steps=1000,
)

timestep = 200

diffusion = GaussianDiffusion(args.diffusion_steps, args.device)

transform_diff = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class_label = torch.tensor([label], device=args.device)

# setup diffusion model
metadata = get_metadata(args.dataset)
model = unets.__dict__[args.arch](
    image_size=metadata.image_size,
    in_channels=metadata.num_channels,
    out_channels=metadata.num_channels,
    num_classes=metadata.num_classes,
).to(args.device)

# load the pre-trained model
print(f"Loading pretrained model from {args.pretrained_ckpt}")
d = fix_legacy_dict(torch.load(args.pretrained_ckpt, map_location=args.device))
dm = model.state_dict()
model.load_state_dict(d, strict=False)

# Sample from the diffusion model
model.eval()

cond_gradients = sample_image(
    model,
    diffusion,
    image_no_transform.to(args.device),
    label,
    timestep,
    args=args
)

normalized_gradients = cond_gradients[0]
normalized_gradients = (normalized_gradients - normalized_gradients.min()) / (normalized_gradients.max() - normalized_gradients.min() + 1e-8)

saliency_diffusion = np.abs(cond_gradients)

saliency_map_diffusion = np.max(saliency_diffusion, axis=0)

saliency_map_diffusion = (saliency_map_diffusion - saliency_map_diffusion.min()) / (saliency_map_diffusion.max() - saliency_map_diffusion.min() + 1e-8)


print(saliency_map_diffusion.shape, saliency_maps[1].shape)

# show images

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 2)
plt.imshow(saliency_maps[0], cmap='viridis')
plt.title(f"Predicted: {class_names[predicted_classes[0]]}")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(saliency_maps[1], cmap='viridis')
plt.title(f"Predicted: {class_names[predicted_classes[1]]}")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(saliency_map_diffusion.transpose(1, 2, 0))
plt.title(f"Diffusion Gradient\nClass: {class_names[label]}")
plt.axis('off')

plt.tight_layout()
plt.savefig(f"/n/fs/visualai-scr/temp_LLP/sofia/llp-work/analysis/website.png")

