# Kaelyn Jefferson
# Project 5: Recognition using Deep Networks -- Question 1
# BUILD WITH: python3 buildAndTrainNetwork.py execute

# import statements
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import cv2
import time


# class definitions
class MyNetwork(nn.Module):

    # initializes a network
    # code sourced from: https://nextjournal.com/gkoehler/pytorch-mnist
    def __init__(self, dropOutRate):
        super(MyNetwork, self).__init__()
        # convol layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # convol layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # convol layer with 0.5 drop out rate
        self.conv2_drop = nn.Dropout2d(p=dropOutRate)

        # fully connected (linear layer) with 50 nodes
        self.fc1 = nn.Linear(320, 50)

        # final fully connected layer with 10 nodes
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # code sourced from: https://nextjournal.com/gkoehler/pytorch-mnist
    def forward(self, x):

        # max pooling layer with 2x2 window and relu fxn applied
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # max pooling layer with 2x2 window and relu fxn applied
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # flattening operation to 1D vec
        x = x.view(-1, 320)

        # relu on the output?
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        # log.softmax function applied to the output
        return F.log_softmax(x)


# function to get MNIST digit data set
# code sourced from: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# code sourced from: https://nextjournal.com/gkoehler/pytorch-mnist
def prepare_data_set(batch_size_train):

    # get the MNIST digit data set
    # batch_size_train = 64
    batch_size_test = 1000

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            # root="data",
            "mnist",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            # root="data",
            "mnist",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )

    # plot the first six examples of the data
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    """
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        print(example_data[i][0])

        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    print(fig)
    plt.show()
    """

    return train_loader, test_loader, example_data, example_targets


# training the model
# code sourced from: https://nextjournal.com/gkoehler/pytorch-mnist
def train_network(
    network, optimizer, log_interval, epoch, train_loader, train_losses, train_counter
):

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )
            # torch.save(network.state_dict(), "network.pt")
            # torch.save(optimizer.state_dict(), "optimizer.pt")
    return train_losses, train_counter


# testing the model
# code sourced from: https://nextjournal.com/gkoehler/pytorch-mnist
def test(network, test_loader, test_losses):

    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_losses


def trainModel(network, train_loader, test_loader, optimizer, n_epochs):

    test_losses = []
    train_losses = []
    train_counter = []

    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    log_interval = 10
    test_losses = test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train_losses, train_counter = train_network(
            network,
            optimizer,
            log_interval,
            epoch,
            train_loader,
            train_losses,
            train_counter,
        )
        test_losses = test(network, test_loader, test_losses)

    """
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    print(fig)
    plt.show()
    """

    return


# main function (yes, it needs a comment too)
def main(argv):

    # alter network structure by these 3 variables:
    # batch size
    # n_epochs
    # drop out rate in networks

    # instantiate 3d array to keep total time of all processes below
    ourArrOfAllTimes = []

    # EXPERIMENT 1: optimize batch_train_size keeping n_epochs = 1 and network = myNetwork()
    # hypothesis: Batch size is to reshuffle the data at every epoch to reduce model overfitting.
    # The higher the batch train size, the bigger the batch, the better the model will do. I think the longer it will take if more in the batch size.
    if argv[1] == "1":

        nums_batch_train_size = [
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
            100,
            105,
        ]

        for i in range(20):
            batch_size_train = nums_batch_train_size[i]
            n_epochs = 1
            network = MyNetwork(0.2)

            random_seed = 42
            torch.backends.cudnn.enabled = False
            torch.manual_seed(random_seed)

            # prepare data set
            (
                train_loader,
                test_loader,
                example_data,
                example_targets,
            ) = prepare_data_set(batch_size_train)

            # Build a network model
            learning_rate = 0.01
            momentum = 0.5

            optimizer = optim.SGD(
                network.parameters(), lr=learning_rate, momentum=momentum
            )

            # training the model: time this
            start = time.time()
            trainModel(network, train_loader, test_loader, optimizer, n_epochs)
            end = time.time()
            totalTime = end - start
            print("TOTAL TIME ELAPSED: ", totalTime)

            ourArrOfAllTimes.append(totalTime)

        # main function code

        for i in range(len(ourArrOfAllTimes)):
            print(
                "batch_size_train = ",
                nums_batch_train_size[i],
                " tot time elapsed: ",
                ourArrOfAllTimes[i],
            )

    # EXPERIMENT 2: optimize n_epochs,  keeping batch_train_size = ? and network = myNetwork()
    # hypothesis: Number of epochss shows how many iterations the model will learn from .
    # Using the best time efficient batch_train_size, the num of epochs should be slower as more training is done.
    # Num of epochs is a tradeoff: more epochs, better model, but slower

    if argv[1] == "2":

        epochs = [1, 2, 3, 4, 5]

        for i in range(len(epochs)):
            batch_size_train = 90  # CHANGE HERE FOR OPTIMAL
            n_epochs = epochs[i]
            network = MyNetwork(0.2)

            random_seed = 42
            torch.backends.cudnn.enabled = False
            torch.manual_seed(random_seed)

            # prepare data set
            (
                train_loader,
                test_loader,
                example_data,
                example_targets,
            ) = prepare_data_set(batch_size_train)

            # Build a network model
            learning_rate = 0.01
            momentum = 0.5

            optimizer = optim.SGD(
                network.parameters(), lr=learning_rate, momentum=momentum
            )

            # training the model: time this
            start = time.time()
            trainModel(network, train_loader, test_loader, optimizer, n_epochs)
            end = time.time()
            totalTime = end - start
            print("TOTAL TIME ELAPSED: ", totalTime)

            ourArrOfAllTimes.append(totalTime)

        # main function code

        for i in range(len(ourArrOfAllTimes)):
            print(
                "epoch size = ",
                epochs[i],
                " tot time elapsed: ",
                ourArrOfAllTimes[i],
            )

    # EXPERIMENT 3: optimize the dropout rate in the networks,  keeping batch_train_size = 90 and n_epochs = 5
    # hypothesis: Dropout layers are used for regularizing the model, or preventing overfitting issues. Dropout layers
    # randomly set some inputs to zero, so maybe the higher dropout, the more processing needed and longer time.
    # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

    if argv[1] == "3":
        dropOutRates = [
            0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ]

        for i in range(len(dropOutRates)):
            batch_size_train = 90
            n_epochs = 5  # CHANGE THIS WHEN WE KNOW
            network = MyNetwork(dropOutRates[i])

            random_seed = 42
            torch.backends.cudnn.enabled = False
            torch.manual_seed(random_seed)

            # prepare data set
            (
                train_loader,
                test_loader,
                example_data,
                example_targets,
            ) = prepare_data_set(batch_size_train)

            # Build a network model
            learning_rate = 0.01
            momentum = 0.5

            optimizer = optim.SGD(
                network.parameters(), lr=learning_rate, momentum=momentum
            )

            # training the model: time this
            start = time.time()
            trainModel(network, train_loader, test_loader, optimizer, n_epochs)
            end = time.time()
            totalTime = end - start
            print("TOTAL TIME ELAPSED: ", totalTime)

            ourArrOfAllTimes.append(totalTime)

        for i in range(len(ourArrOfAllTimes)):
            print(
                "dropout rate = ",
                dropOutRates[i],
                " tot time elapsed: ",
                ourArrOfAllTimes[i],
            )
    return


if __name__ == "__main__":
    main(sys.argv)
