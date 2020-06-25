from OneConvOneFcNet import *
from OneFcNet import *
from TwoConvTwoFcNet import *
from load_cifar10 import *
from TwoFcNet import *
import torch.optim as optim
import torchvision.models as models


import torch
import torchvision
import torchvision.transforms as transforms


def run():
    torch.multiprocessing.freeze_support()
    nets = [ TwoFcNet(),
             OneFcNet(),
             TwoConvTwoFcNet(),
             models.vgg16()]
#             OneConvOneFcNet(),
#              ]

    criterion = nn.CrossEntropyLoss()

    cifar_10_dir = 'cifar-10-batches-py'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=1)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for net in nets:
        print("running on model" , net.__module__)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                # get the inputs
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
                if i % 2000 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

if __name__ == '__main__':
    run()



