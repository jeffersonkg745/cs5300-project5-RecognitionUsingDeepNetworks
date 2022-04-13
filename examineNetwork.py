# Kaelyn Jefferson
# Project 5: Recognition using Deep Networks
# Question 2: Analyze the network and see how it processes the data
# BUILD WITH: python3 examineNetwork.py

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


# sub class of MyNetwork in buildAndTrainNetwork.py
# python subclass help: https://stackoverflow.com/questions/1607612/python-how-do-i-make-a-subclass-from-a-superclass
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
        return x


#  Read in the trained network and print the model
def printModel():
    network_model = torch.load("networkSaved.pt")
    network_model.eval()
    print(network_model)
    return network_model


def getWeightsOfFirstLayer(model):

    fig = plt.figure()
    for i in range(10):
        print("size of weight #", i)
        print(model.conv1.weight[i, 0].shape)
        print("\n")

        print("filter weight #", i)
        print(model.conv1.weight[i, 0])
        print("\n")

        # plot data: sourced matplotlib docs
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(model.conv1.weight[i, 0].detach().numpy())
        plt.title("Filter #: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    print(fig)
    plt.show()
    return


# opencv docs
def applyFilters(model):
    with torch.no_grad():

        fig = plt.figure()
        for i in range(10):

            firstImage = cv2.imread("data/numsjpg/image0.jpeg")

            # convert tensor to numpy for filter2d fxn : https://www.codegrepper.com/code-examples/python/convert+a+tensor+to+numpy+array
            # sourced code on docs: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
            kernel = model.conv1.weight[i, 0].numpy()
            dst = cv2.filter2D(firstImage, -1, kernel)

            # plot data: sourced matplotlib docs
            plt.subplot(4, 5, i + 1)
            plt.tight_layout()
            plt.imshow(kernel)
            plt.title("Filter #: {}".format(i))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(4, 5, i + 11)
            plt.tight_layout()
            plt.imshow(dst)
            plt.title("Img #: {}".format(i))
            plt.xticks([])
            plt.yticks([])

        print(fig)
        plt.show()

    return


def createNewSubmodel():
    # create object
    new_model = Submodel()

    # load state dictionary
    new_model.load_state_dict(
        torch.load(
            "/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/cs5300-project5-RecognitionUsingDeepNetworks/network.pt"
        )
    )
    new_model.eval()

    # PROB: should have 20 channels 4x4 in size??

    # apply the model to the first example image: use 2nd layer output for submission
    with torch.no_grad():

        fig = plt.figure()
        for i in range(10):

            firstImage = cv2.imread("data/numsjpg/image0.jpeg")

            # convert tensor to numpy for filter2d fxn : https://www.codegrepper.com/code-examples/python/convert+a+tensor+to+numpy+array
            # sourced code on docs: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
            kernel = new_model.conv2.weight[i, 0].numpy()
            dst = cv2.filter2D(firstImage, -1, kernel)

            # plot data: sourced matplotlib docs
            plt.subplot(4, 5, i + 1)
            plt.tight_layout()
            plt.imshow(kernel)
            plt.title("Filter #: {}".format(i))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(4, 5, i + 11)
            plt.tight_layout()
            plt.imshow(dst)
            plt.title("Img #: {}".format(i))
            plt.xticks([])
            plt.yticks([])

        print(fig)
        plt.show()

    return


# main function (yes, it needs a comment too)
def main(argv):

    # Part 2
    model = printModel()

    # Part 2A
    # getWeightsOfFirstLayer(model)

    # Part 2B
    # applyFilters(model)

    # Part 3C
    createNewSubmodel()

    return


if __name__ == "__main__":
    main(sys.argv)
