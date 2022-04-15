# Kaelyn Jefferson
# Project 5: Recognition using Deep Networks -- Question 1F and 1G


# import statements
from lib2to3.pytree import convert
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from buildAndTrainNetwork import MyNetwork
import buildAndTrainNetwork
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import cv2


# use with part 1f
def runOnTestSet(network_model, example_data, example_targets):

    # for i in range(len(example_data)):
    # example_data[i][0] = torchvision.transforms.Grayscale(example_data[i][0])
    # example_data = example_data.np().transpose(1, 2, 0)

    # print the prediction and the number below it
    with torch.no_grad():

        # PROBLEM HERE: need to convert example_data to gray
        output = network_model(example_data)

        # plot the data of first 9
        fig = plt.figure()

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
            plt.title(
                "Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item())
            )
            print(
                "Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item())
            )
            print("Index is: {}".format(i))
            print("Ground Truth: {}".format(example_targets[i]))
            print("\n")
            plt.xticks([])
            plt.yticks([])

        print(fig)
        plt.show()

    return


# use with 1G
# folder structure shown here: https://stackoverflow.com/questions/49073799/pytorch-testing-with-torchvision-datasets-imagefolder-and-dataloader
# code sourced from: https://www.kaggle.com/code/leifuer/intro-to-pytorch-loading-image-data/notebook
# code sourced from: https://pytorch.org/vision/0.8/datasets.html#imagefolder
def prepare_custom_data_set():

    # make our own data set
    dataset = torchvision.datasets.ImageFolder(
        root="/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/cs5300-project5-RecognitionUsingDeepNetworks/data/custom_nums",
        transform=torchvision.transforms.ToTensor(),
    )
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # plot the first six examples of the data
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

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

    return example_data, example_targets


# main function (yes, it needs a comment too)
def main(argv):

    # Question 1F
    # 1. read the network
    network_model = torch.load("networkSaved.pt")
    network_model.eval()

    # 2. get the example data again
    (
        train_loader,
        test_loader,
        example_data,
        example_targets,
    ) = buildAndTrainNetwork.prepare_data_set()

    # 3. run the first 10 examples in the test set
    runOnTestSet(network_model, example_data, example_targets)

    # Question 1G
    # 1. crop and resize photos using magik from command line to 28x28
    # 2. read images and convert them to grayscale, sourced from: https://medium.com/analytics-vidhya/create-your-own-real-image-dataset-with-python-deep-learning-b2576b63da1e

    path = "/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/cs5300-project5-RecognitionUsingDeepNetworks/data/custom_nums_all"

    for img in os.listdir(path):
        pic = cv2.imread(os.path.join(path, img), 0)

        # if the number of channels is 3 for bgr, then convert to gray scale
        # sourced code from https://stackoverflow.com/questions/19062875/how-to-get-the-number-of-channels-from-an-image-in-opencv-2

        if len(pic.shape) == 3:
            pic = cv2.resize(pic, (28, 28), interpolation=cv2.INTER_AREA)
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            pic = np.invert(pic)
        cv2.imwrite(os.path.join(path, img), pic)

    # 3. get the data from our custom data set (hand written number images)
    (
        custom_example_data,
        custom_example_targets,
    ) = prepare_custom_data_set()

    # 4. run the first 10 examples in the test set
    runOnTestSet(network_model, custom_example_data, custom_example_targets)

    return


if __name__ == "__main__":
    main(sys.argv)
