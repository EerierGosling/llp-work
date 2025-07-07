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

parser = argparse.ArgumentParser()

adversarial = "2025-07-03 14:34:41"
non_adversarial = "2025-07-03 14:34:43"

website = True

if website:
    parser.add_argument('--file_name', type=str, required=True)

args = parser.parse_args()

time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

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
)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

if website:
    image_no_transform = Image.open(args.file_name).convert('RGB')
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


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(np.array(image_no_transform))
plt.title(f"Original: {class_names[label]}")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(saliency_maps[0])
plt.title(f"Predicted: {class_names[predicted_classes[i]]}")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(saliency_maps[1])
plt.title(f"Predicted: {class_names[predicted_classes[i]]}")
plt.axis('off')

plt.tight_layout()
plt.savefig(f"analysis/{time}.png")