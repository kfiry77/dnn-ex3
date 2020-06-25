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
from trains import Task

def run():
    torch.multiprocessing.freeze_support()
    nets = [ TwoConvTwoFcNet(),
             OneConvOneFcNet(),
             TwoFcNet(),
             OneFcNet(),
             models.vgg16()]
    nets_active = [True, False, False, False, False]

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

    for i in range(0, len(nets)):
        if not nets_active[i]:
            continue
        net = nets[i]
        model_name = net.__module__
        print("running on model", model_name)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        task = Task.init(project_name="MonitorsTest", task_name=model_name)

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
            print('Finsihed Training')

            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))

if __name__ == '__main__':
    run()



