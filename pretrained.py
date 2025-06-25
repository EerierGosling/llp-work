import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch.nn as nn
import wandb
import datetime

config={
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "batch_size": 32,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
    "epochs": 100,
}

model = models.resnet34(weights='IMAGENET1K_V1')
model.maxpool = nn.Identity()
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)


model.eval()

num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs, device):
    model.train()
    
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        test_acc = evaluate_model(model, testloader, device=device)

        wandb.log({"train_acc": train_correct / train_total, "test_acc": test_acc})
        print("logged")

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total



if __name__ == "__main__":
    wandb.init(
        project="classfier-cifar10-resnet34",
        name=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        config=config,
    )
    
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    train_model(model, trainloader, testloader, criterion, optimizer, num_epochs=wandb.config.epochs, device=device)