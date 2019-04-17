import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import matplotlib.ticker as ticker

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 12, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(12 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

# Hyper Parameters
EPOCH = 20

transform = transforms.Compose([
transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)

cnn1 = CNN()
cnn2 = CNN()
print(cnn1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn1.parameters(), lr=0.001)
optimizer2 = optim.Adam(cnn2.parameters(), lr=0.0001)

x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
SGD_training_accuracy=[]
SGD_validation_accuracy=[]
SGD_training_loss=[]
SGD_validation_loss=[]
Adam_training_accuracy=[]
Adam_validation_accuracy=[]
Adam_training_loss=[]
Adam_validation_loss=[]

# training and testing of SGD
for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    correct = 0
    total = 0
    running_loss = 0.0
    count=0.

    with torch.no_grad():
        for data in trainloader:
            count+=1
            images, labels = data
            outputs = cnn1(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    SGD_training_loss.append(running_loss/count)
    accuracy=1. * correct / total
    SGD_training_accuracy.append(accuracy)
    print('Accuracy of the network using SGD optimization algorithm on the train set: %f %%' % (
            100. * correct / total))
    print(running_loss/count)

    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = cnn1(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    SGD_validation_loss.append(running_loss/count)
    accuracy = 1. * correct / total
    SGD_validation_accuracy.append(accuracy)
    print('Accuracy of the network using SGD optimization algorithm on the test set: %f %%' % (
            100. * correct / total))
    print(running_loss/count)

# training and testing of Adam
for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer2.zero_grad()

        # forward + backward + optimize
        outputs = cnn2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    correct = 0
    total = 0
    running_loss = 0.0
    count=0.

    with torch.no_grad():
        for data in trainloader:
            count+=1
            images, labels = data
            outputs = cnn2(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    Adam_training_loss.append(running_loss/count)
    accuracy=1. * correct / total
    Adam_training_accuracy.append(accuracy)
    print('Accuracy of the network using Adam optimization algorithm on the train set: %f %%' % (
            100. * correct / total))
    print(running_loss/count)

    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = cnn2(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    Adam_validation_loss.append(running_loss/count)
    accuracy = 1. * correct / total
    Adam_validation_accuracy.append(accuracy)
    print('Accuracy of the network using Adam optimization algorithm on the test set: %f %%' % (
            100. * correct / total))
    print(running_loss/count)

print("Accuracy of CNN1: %d"% SGD_validation_accuracy[EPOCH-1])
print("Accuracy of CNN2: %d"% Adam_validation_accuracy[EPOCH-1])

plt.xlabel("epoch")
plt.ylabel("SGD Training Accuracy")
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.plot(x, SGD_training_accuracy, marker='o')
plt.show()
plt.xlabel("epoch")
plt.ylabel("SGD Validation Accuracy")
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.plot(x, SGD_validation_accuracy, marker='o')
plt.show()
plt.xlabel("epoch")
plt.ylabel("SGD Training Loss")
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.plot(x, SGD_training_loss, marker='o')
plt.show()
plt.xlabel("epoch")
plt.ylabel("SGD Validation Loss")
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.plot(x, SGD_validation_loss, marker='o')
plt.show()


plt.xlabel("epoch")
plt.ylabel("Adam Training Accuracy")
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.plot(x, Adam_training_accuracy, marker='o')
plt.show()
plt.xlabel("epoch")
plt.ylabel("Adam Validation Accuracy")
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.plot(x, Adam_validation_accuracy, marker='o')
plt.show()
plt.xlabel("epoch")
plt.ylabel("Adam Training Loss")
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.plot(x, Adam_training_loss, marker='o')
plt.show()
plt.xlabel("epoch")
plt.ylabel("Adam Validation Loss")
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.plot(x, Adam_validation_loss, marker='o')
plt.show()

