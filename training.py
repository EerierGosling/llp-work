# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import argparse

num_epochs = 5

def reset_csv():
    with open('results.csv', 'w') as f:
        f.write('epoch,' + ','.join([str(i) for i in range(1, num_epochs+1)]) + '\n')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

learning_rate_options = [0.001, 0.005, 0.01, 0.05, 0.1]


if __name__ == '__main__':

    net = Net()

    criterion = nn.CrossEntropyLoss()

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int, required=True)
    job_id = parser.parse_args().job_id

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    learning_rate = learning_rate_options[job_id]

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    accuracy = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        # testing the model
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy.append(correct/total)

    with open('results.csv', 'a') as f:
        f.write(f'{learning_rate},' + ','.join([str(a) for a in accuracy]) + '\n')

    print('Finished Training')

    PATH = f'./trained-models/cifar_net-lr_{learning_rate}.pth'
    torch.save(net.state_dict(), PATH)

    print("saved")

    # dataiter = iter(testloader)
    # images, labels = next(dataiter)

    # # print images
    # img_grid = torchvision.utils.make_grid(images)
    # # Convert from CHW to HWC format and denormalize
    # img_grid = img_grid.permute(1, 2, 0)
    # img_grid = img_grid * 0.5 + 0.5  # denormalize from [-1, 1] to [0, 1]
    # plt.imshow(img_grid)
    # plt.show()
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    
    # net = Net()
    # net.load_state_dict(torch.load(PATH, weights_only=True))

    # outputs = net(images)

    # _, predicted = torch.max(outputs, 1)

    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
    #                           for j in range(4)))
    
    # correct = 0
    # total = 0
    # # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         # calculate outputs by running images through the network
    #         outputs = net(images)
    #         # the class with the highest energy is what we choose as prediction
    #         _, predicted = torch.max(outputs, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # # prepare to count predictions for each class
    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}

    # # again no gradients needed
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predictions = torch.max(outputs, 1)
    #         # collect the correct predictions for each class
    #         for label, prediction in zip(labels, predictions):
    #             if label == prediction:
    #                 correct_pred[classes[label]] += 1
    #             total_pred[classes[label]] += 1


    # # print accuracy for each class
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')