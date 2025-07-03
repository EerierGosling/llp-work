import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import wandb
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, required=True)
parser.add_argument('--epsilon', type=float, required=True)
parser.add_argument('--adversarial_ratio', type=float, required=True)
parser.add_argument('--adversarial_training', type=bool, required=True)

args = parser.parse_args()

config={
    "learning_rate": args.learning_rate,
    "weight_decay": args.weight_decay,
    "batch_size": 32,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
    "epochs": 100,
    "epsilon": args.epsilon,
    "adversarial_ratio": args.adversarial_ratio,
    "warmup_epochs": 10,
    "adversarial_training": args.adversarial_training,
}


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout_conv(x)
        
        x = F.relu(self.bn3(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.dropout_conv(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
if __name__ == '__main__':

    model = models.resnet34()
    model.maxpool = nn.Identity()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    
    print("starting")
    name = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    wandb.init(
        project="classfier-cifar10-adversarial",
        name=name,
        config=config,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=2)

    optimizer = optim.AdamW(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=wandb.config.epochs)

    accuracy = []

    for epoch in range(wandb.config.epochs):
        epoch_loss = 0.0
        train_total = 0
        train_correct = 0
        adv_correct = 0
        model.train()
        
        current_adv_ratio = 0.0 if epoch < wandb.config.warmup_epochs else wandb.config.adversarial_ratio

        if epoch >= wandb.config.warmup_epochs and wandb.config.adversarial_training:
            ramp_epochs = 10
            current_adv_ratio = min(wandb.config.adversarial_ratio, (epoch - wandb.config.warmup_epochs) / ramp_epochs * wandb.config.adversarial_ratio)
        else:
            current_adv_ratio = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            clean_outputs = model(inputs)
            clean_loss = criterion(clean_outputs, labels)
            
            adv_outputs = clean_outputs
            
            if current_adv_ratio > 0 and wandb.config.adversarial_training:
                total_adv_loss = 0
                attack_strengths = [wandb.config.epsilon * 0.5, wandb.config.epsilon, wandb.config.epsilon * 1.5]
                
                for eps in attack_strengths:
                    inputs_copy = inputs.clone().detach().requires_grad_(True)
                    adv_outputs_temp = model(inputs_copy)
                    adv_loss_temp = criterion(adv_outputs_temp, labels)
                    model.zero_grad()
                    adv_loss_temp.backward()
                    data_grad = inputs_copy.grad.data
                    
                    adv_inputs = fgsm_attack(inputs_copy, eps, data_grad)
                    adv_outputs = model(adv_inputs)
                    total_adv_loss += criterion(adv_outputs, labels)
                
                total_adv_loss /= len(attack_strengths)
                total_loss = (1 - current_adv_ratio) * clean_loss + current_adv_ratio * total_adv_loss
            else:
                total_loss = clean_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            # Calculate accuracy on clean examples
            _, predicted = torch.max(clean_outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Calculate accuracy on adversarial examples
            _, adv_predicted = torch.max(adv_outputs, 1)
            adv_correct += (adv_predicted == labels).sum().item()
        
        model.eval()
        # testing the model
        test_correct = 0
        test_total = 0
        adv_test_correct = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                

        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)          

            images.requires_grad = True
            outputs = model(images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = images.grad.data
            
            adv_images = fgsm_attack(images, wandb.config.epsilon, data_grad)
            with torch.no_grad():
                adv_outputs = model(adv_images)
                _, adv_predicted = torch.max(adv_outputs, 1)
                adv_test_correct += (adv_predicted == labels).sum().item()

        to_log = {
            "train_acc": train_correct / train_total,
            "train_adv_acc": adv_correct / train_total,
            "test_acc": test_correct / test_total,
            "test_adv_acc": adv_test_correct / test_total,
            "loss": epoch_loss / len(trainloader)
        }

        if current_adv_ratio > 0 and wandb.config.adversarial_training:
            to_log["adversarial_ratio"] = current_adv_ratio

        wandb.log(to_log)
        scheduler.step()
        epoch_loss = 0.0

    PATH = f'./trained-models/{name}.pth'
    torch.save(model.state_dict(), PATH)

    wandb.finish()

    print("saved")