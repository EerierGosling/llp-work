import torch
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision import transforms
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

name = "2025-07-01 14:07:25"

model = models.resnet101(num_classes=10)

PATH = f'./trained-models/{name}.pth'
model.load_state_dict(torch.load(PATH))

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_dataset = datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

random_idx = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[random_idx]
input_batch = image.unsqueeze(0)
input_batch = input_batch.to(device)
input_batch.requires_grad = True


output = model(input_batch)

_, predicted_idx = torch.max(output, 1)
predicted_class = predicted_idx.item()

print("predicted:" + class_names[label])
print("actual:" + class_names[predicted_class])

score = output[0, predicted_class]

gradients = torch.autograd.grad(outputs=score, inputs=input_batch)[0]

saliency = torch.abs(gradients)

saliency_map = torch.max(saliency, dim=1)[0].squeeze().cpu().numpy()

saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
plt.title(f"Original: {class_names[label]}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap='hot')
plt.title(f"Predicted: {class_names[predicted_class]}")
plt.axis('off')

plt.tight_layout()
plt.savefig('saliency_map.png')
plt.show()