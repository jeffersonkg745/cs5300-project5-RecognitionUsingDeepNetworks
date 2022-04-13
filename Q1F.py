# Question 1F: read network + run on test set

# import statements
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from buildAndTrainNetwork import MyNetwork


# import numpy as np
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Question 1F
# read the network
network_model = torch.load("networkSaved.pt")  # maybe send this to test()
network_model.eval()


# run the model on the first 10 examples of the test set
batch_size_test = 1000

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
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


fig = plt.figure()
for i in range(10):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(test_loader[i][0], cmap="gray", interpolation="none")
    plt.title(
        # uses the data from the test set?
        "Prediction: {}".format(network_model.data.max(1, keepdim=True)[1][i].item())
    )
    plt.xticks([])
    plt.yticks([])
print(fig)
