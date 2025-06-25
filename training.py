# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pretrained
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import argparse
import os
import wandb
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', type=int, required=True)
job_id = parser.parse_args().job_id

config={
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "batch_size": 32,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
    "epochs": 40,
}

wandb.init(
    project="classfier-cifar10",
    name=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    config=config,
)


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
    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    trainset = pretrained.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=2)

    testset = pretrained.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=2)

    optimizer = optim.AdamW(net.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=wandb.config.epochs)

    accuracy = []

    for epoch in range(wandb.config.epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        train_total = 0
        train_correct = 0
        net.train()
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        net.eval()
        # testing the model
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move data to GPU
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        wandb.log({"train_acc": train_correct / train_total, "test_acc": test_correct / test_total, "loss": epoch_loss / len(trainloader)})
        scheduler.step()
        epoch_loss = 0.0

    PATH = f'./trained-models/cifar_net-lr_{wandb.config.learning_rate}.pth'
    torch.save(net.state_dict(), PATH)

    wandb.finish()

    print("saved")