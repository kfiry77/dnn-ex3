from OneConvOneFcNet import *
from OneFcNet import *
from TwoConvTwoFcNet import *
from load_cifar10 import *
from TwoFcNet import *
import torch.optim as optim
import torchvision.models as models
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from trains import Task, Logger

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

def get_model_params(model):
    p_aggregate = None
    for param in model.parameters():
        t = torch.flatten(param.data)
        if p_aggregate is None:
            p_aggregate = t
        else:
            p_aggregate = torch.cat((p_aggregate, t))
    return p_aggregate

def evaluate_wieghts(cur_weights, prev_weights):
    return torch.mean(torch.abs(cur_weights - prev_weights))

nets = [models.resnet50(),
        TwoConvTwoFcNet(),
        OneConvOneFcNet(),
        TwoFcNet(),
        OneFcNet(),
        models.vgg16()] # sorry my computer couldn't cop with it.

nets_active = [True, True, True, True, True]

def run():
    torch.multiprocessing.freeze_support()
    for i in range(0, len(nets)):
        if not nets_active[i]:
            continue
        net = nets[i]
        model_name = net.__module__
        print("running on model", model_name)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        task = Task.init(project_name="Ex3_1", task_name=model_name)

        # q1 - evaluate model before we start.
        evaluate_model(0, net, task)

        prev_weights = get_model_params(net)

        for epoch in range(1, 15):
            running_loss = 0.0

            train_loss = []
            for j, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # check the weights and report it.
                cur_weights = get_model_params(net)
                w = evaluate_wieghts(prev_weights, cur_weights)
                task.get_logger().report_scalar(
                    "weights", "aver", iteration=epoch * len(trainloader) + j, value=w)

                prev_weights = cur_weights

                # print statistics
                train_loss.append(loss.item())
                running_loss += loss.item()
                if j % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, j + 1, running_loss / 2000))
                    task.get_logger().report_scalar(
                        "train", "loss", iteration=(epoch * len(trainloader) + j), value=running_loss / 2000)
                    running_loss = 0.0

            print(f'Finsihed Training loss={np.mean(train_loss)}')
            task.get_logger().report_scalar(
                "loss", "train", iteration=epoch, value=np.mean(train_loss))
            evaluate_model(epoch, net, task)

        task.close()

def evaluate_model(epoch, net, task):
    test_loss = []
    test_accuracy = []
    for i, (data, labels) in enumerate(testloader):
        # pass data through network
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        test_loss.append(loss.item())
        test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
    print(f'epoch: {epoch} test loss: {np.mean(test_loss)}, test accuracy: {np.mean(test_accuracy)}')
    task.get_logger().report_scalar(
        "loss", "test", iteration=epoch, value=np.mean(test_loss))
    task.get_logger().report_scalar(
        "accuracy", "test", iteration=epoch, value=(np.mean(test_accuracy)))

if __name__ == '__main__':
    run()
