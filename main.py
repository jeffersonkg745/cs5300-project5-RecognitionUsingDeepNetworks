# Kaelyn Jefferson
# Project 5: Recognition using Deep Networks

# import statements
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

# import numpy as np
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# class definitions
class MyNetwork(nn.Module):

    # initializes a network
    # code sourced from: https://nextjournal.com/gkoehler/pytorch-mnist
    def __init__(self):
        super(MyNetwork, self).__init__()
        # convol layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # convol layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # convol layer with 0.5 drop out rate
        self.conv2_drop = nn.Dropout2d(p=0.5)

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

        # flattening operation?
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
def prepare_data_set():

    # get the MNIST digit data set
    batch_size_train = 64
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

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    print(fig)
    plt.show()

    # Look up examples that make use of the pyplot subplot method to create a grid of plots.

    return train_loader, test_loader


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
            torch.save(network.state_dict(), "network.pt")
            torch.save(optimizer.state_dict(), "optimizer.pt")
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


# main function (yes, it needs a comment too)
def main(argv):

    # Question 1B
    # make the network repeatable at start of main fxn
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # handle any command line arguments in argv
    if argv[1] == "execute":

        # Question 1A
        train_loader, test_loader = prepare_data_set()

        # Question 1C
        # Build a network model
        learning_rate = 0.01
        momentum = 0.5
        network = MyNetwork()
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

        # Question 1D
        # training the model
        test_losses = []
        train_losses = []
        train_counter = []
        n_epochs = 5
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

        fig = plt.figure()
        plt.plot(train_counter, train_losses, color="blue")
        plt.scatter(test_counter, test_losses, color="red")
        plt.legend(["Train Loss", "Test Loss"], loc="upper right")
        plt.xlabel("number of training examples seen")
        plt.ylabel("negative log likelihood loss")
        print(fig)
        plt.show()

    elif argv[1] == "import":
        # call something else here
        print("fxn not created yet")

    # main function code
    return


if __name__ == "__main__":
    main(sys.argv)
