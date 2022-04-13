# Kaelyn Jefferson
# Project 5: Recognition using Deep Networks
# Question 3: digit embedding space
# BUILD WITH: python3 digitEmbeddingSpace.py

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
from buildAndTrainNetwork import MyNetwork

# Model subclass of the MyNetwork class that has all but the output layer
class Submodel(MyNetwork):
    def __init__(self):
        super().__init__()

    # override the forward method
    def forward(self, x):
        # relu on max pooled results of conv1
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # relu on max pooled results of dropout of conv2
        x = F.relu(
            F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)
        )  # try modifying forward to return after first layer
        """
        # flattening operation to 1D vec
        x = x.view(-1, 320)

        # relu on the output?
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        # log.softmax function applied to the output
        return F.log_softmax(x)
        """
        return x


# main function (yes, it needs a comment too)
def main(argv):

    # read in the network
    network_model = torch.load("networkSaved.pt")
    network_model.eval()

    # make a submodel that includes everything but the output layer
    model = Submodel()

    # Part 3A

    return


if __name__ == "__main__":
    main(sys.argv)
