import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import wandb
import uuid
import os
from torchattacks import PGD

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, required=True)
parser.add_argument('--epsilon', type=float, required=True)
parser.add_argument('--common_test_epsilons', type=str, required=True)
parser.add_argument('--adversarial_ratio', type=float, required=True)
parser.add_argument('--adversarial_training', action="store_true")
parser.add_argument('--run_name', type=str, required=True)

args = parser.parse_args()

common_test_epsilons = [float(eps) for eps in args.common_test_epsilons.split(',')]

if args.epsilon not in common_test_epsilons:
    common_test_epsilons.append(args.epsilon)

config={
    "learning_rate": args.learning_rate,
    "weight_decay": args.weight_decay,
    "batch_size": 32,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
    "epochs": 100,
    "epsilon": args.epsilon,
    "common_test_epsilons": common_test_epsilons,
    "adversarial_ratio": args.adversarial_ratio,
    "warmup_epochs": 10,
    "adversarial_training": args.adversarial_training,
    "resnet_type": "resnet34",
}

# Device setup
device = "cuda"
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


        
if __name__ == '__main__':

    model = models.resnet34()
    model.maxpool = nn.Identity()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    
    print("starting")
    wandb.init(
        project="classfier-cifar10-adversarial",
        config=config,
    )
    model.to(device)

    atk = PGD(model, eps=wandb.config.epsilon, alpha=2/225, steps=5, random_start=True)
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

                adv_inputs = atk(inputs, labels)
                adv_outputs = model(adv_inputs)
                adv_loss = criterion(adv_outputs, labels)

                total_loss = (1 - current_adv_ratio) * clean_loss + current_adv_ratio * adv_loss
            else:
                total_loss = clean_loss


            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            _, predicted = torch.max(clean_outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if current_adv_ratio > 0 and wandb.config.adversarial_training:
                _, adv_predicted = torch.max(adv_outputs, 1)
                adv_correct += (adv_predicted == labels).sum().item()
        
        model.eval()
        
        # testing the model
        test_correct = 0
        test_total = 0
        adv_test_correct = {epsilon: 0 for epsilon in common_test_epsilons}

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
            
            for epsilon in common_test_epsilons:
                adv_images = atk(images, labels)
                
                with torch.no_grad():
                    adv_outputs = model(adv_images)
                    _, adv_predicted = torch.max(adv_outputs, 1)
                    adv_test_correct[epsilon] += (adv_predicted == labels).sum().item()

        to_log = {
            "train_acc": train_correct / train_total,
            "train_adv_acc": adv_correct / train_total,
            "test_acc": test_correct / test_total,
            "loss": epoch_loss / len(trainloader)
        }

        for epsilon, count in adv_test_correct.items():
            to_log[f"test_adv_acc_{epsilon}"] = count / test_total

        if current_adv_ratio > 0 and wandb.config.adversarial_training:
            to_log["adversarial_ratio"] = current_adv_ratio

        wandb.log(to_log)
        scheduler.step()
        epoch_loss = 0.0

        epoch_path = f'/n/fs/visualai-scr/temp_LLP/sofia/llp-work/trained-models/{"adversarial" if wandb.config.adversarial_training else "non_adversarial"}/{wandb.config.resnet_type}/{wandb.config.epsilon}/{args.run_name}.pth'
        os.makedirs(os.path.dirname(epoch_path), exist_ok=True)
        torch.save(model.state_dict(), epoch_path)

    PATH = f'/n/fs/visualai-scr/temp_LLP/sofia/llp-work/trained-models/{"adversarial" if wandb.config.adversarial_training else "non_adversarial"}/{wandb.config.resnet_type}/{wandb.config.epsilon}/{args.run_name}.pth'
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(model.state_dict(), PATH)

    wandb.finish()

    print("saved")