# Kaelyn Jefferson
# Project 5: Recognition using Deep Networks
# Question 3: digit embedding space
# BUILD WITH: python3 digitEmbeddingSpace.py

# import statements
from ctypes import sizeof
from distutils.command.build import build
from tkinter.dnd import dnd_start
from cv2 import compare
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
from Q1FandG import runOnTestSet
from buildAndTrainNetwork import MyNetwork
from os import listdir
from PIL import Image
import csv
from subModel import secondSubModel
import buildAndTrainNetwork
import Q1FandG
import PIL

# class for network
class Submodel(nn.Module):

    # initializes a network
    # code sourced from: https://nextjournal.com/gkoehler/pytorch-mnist
    def __init__(self):
        super(Submodel, self).__init__()
        # convol layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # convol layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # convol layer with 0.5 drop out rate
        self.conv2_drop = nn.Dropout2d(p=0.5)

        # fully connected (linear layer) with 50 nodes
        self.fc1 = nn.Linear(320, 50)

        # final fully connected layer with 10 nodes
        # self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # code sourced from: https://nextjournal.com/gkoehler/pytorch-mnist
    def forward(self, x):

        # max pooling layer with 2x2 window and relu fxn applied
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # max pooling layer with 2x2 window and relu fxn applied
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # flattening operation to 1D vec
        x = x.view(-1, 320)

        # relu on the output
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        # log.softmax function applied to the output
        return F.log_softmax(x)


# source: Dr. Maxwell sample code
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# creates the greek symbol set given the sample images
def createGreekSymbolSet(csvSymbolData, csvSymbolCategories, greekFolder):

    # resets the csv file when you want to create a new greek symbol set
    with open(
        csvSymbolData,
        "w",
    ) as f:
        writer = csv.writer(f)
        writer.writerow("Data for Greek Symbol data set:")
    with open(
        csvSymbolCategories,
        "w",
    ) as f:
        writer = csv.writer(f)
        writer.writerow("Categories for Greek Symbol data set:")

    # source: Dr. Maxwell sample code
    greek_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            greekFolder,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    GreekTransform(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=27,
        shuffle=False,
    )

    # plot the first six examples of the data
    examples = enumerate(greek_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    for i in range(27):
        intensity_values = []
        for row in example_data[i][0]:
            for each in row:
                intensity_values.append(each.item())
        sample_fname, _ = greek_loader.dataset.samples[i]

        # writes to csv with values in
        with open(
            csvSymbolData,
            "a",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(intensity_values)

        # put values in a dictionary here
        categoryDict = {"alpha": 0, "beta": 1, "gamma": 2}
        subStringOfName = sample_fname.split("/")[1]
        if subStringOfName not in categoryDict:
            categoryDict[subStringOfName] = len(categoryDict)

        # writes each line's symbol categories
        with open(
            csvSymbolCategories,
            "a",
            newline="",
        ) as f:
            writer = csv.writer(f)
            num = categoryDict.get(subStringOfName)
            writer.writerow(str(num))

    return greek_loader, example_data, example_targets


# creates the truncated model from the old network that terminates at the dense layer with 50 outputs
def createTruncModel(example_data, example_targets):

    # read in the trained model (from main)
    # build a new model from the old network that terminates at the Dense layer with 50 outputs
    # create object
    new_model = Submodel()

    # read in trained model
    # truncated model needs boolean for strict shown here: https://pytorch.org/tutorials/beginner/saving_loading_models.html#warmstarting-model-using-parameters-from-a-different-model
    new_model.load_state_dict(
        torch.load(
            "/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/cs5300-project5-RecognitionUsingDeepNetworks/network.pt",
        ),
        strict=False,
    )
    new_model.eval()
    # print(new_model)  # shows overall output features is 50

    output = new_model(example_data)
    print(
        "Total output shows 50: {}".format(len(output.data[0]))
    )  # first example image data shows 50

    return


# projects the greek symbols into the embedding space
def projectGreekSymbols(trunc_network, csvsymboldata):

    # get data from csv file
    with open(
        csvsymboldata,
        newline="",
    ) as f:
        reader = csv.reader(f, delimiter=",")

        # keeps array of image intensities
        totalArr = []

        row_count = 0
        row_pointer = 0
        for row in reader:
            if row_pointer == 0:
                row_pointer += 1
                continue
            row = np.reshape(row, (28, 28))
            totalArr.append(row)
        print(len(totalArr))  # 27 is correct

    return totalArr


# compute the distances in the embedding space
def computeDistEmbeddingSpace(network, totalArr, csvsymboldata):

    with open(
        csvsymboldata,
        newline="",
    ) as f:
        reader = csv.reader(f, delimiter=",")

        AlphaBetaGammaImagesData = []

        row_count = 0
        row_pointer = 0
        for row in reader:
            if row_pointer == 0:
                row_pointer += 1
                continue
            row = np.reshape(row, (28, 28))

            if row_pointer == 1 or row_pointer == 10 or row_pointer == 19:

                AlphaBetaGammaImagesData.append(row)

            # increment pointer to look at next row of data in csv
            row_pointer += 1
        print("alpha beta gam: ", len(AlphaBetaGammaImagesData))  # should be 3
        print("total: ", len(totalArr))

        # compute sum squared difference here between each alpha, beta, gamma and the 27 in totalArr
        # 3 = for each example image: alpha, beta, gamma
        for x in range(3):

            currentImage = AlphaBetaGammaImagesData[x]

            # for each image overall (27 total)
            for y in range(27):
                SSD = 0
                comparedImage = totalArr[y]

                # go through image coordinates here
                for i in range(27):
                    for j in range(27):

                        # unsure how to convert tensor value to image space
                        # divide by 255 to get values between 0 and 1
                        ourImageValue = np.float32(currentImage[i][j]) / 255
                        comparedImageValue = np.float32(comparedImage[i][j]) / 255

                        # do SSD here as distance metric
                        difference = ourImageValue - comparedImageValue
                        SSD = SSD + (difference * difference)

                # image 0 is alpha example used
                # image 1 is beta example used
                # image 2 is gamma example used
                # images 0 to 9 are alpha
                # images 10 to 18 are beta
                # images 19 to 27 are gamma
                print("Using image ", x, ", the SSD with image ", y, "is: ", SSD)

    return


# using my own greek symbol data, see how they match in the embedding space
def ownGreekSymbolData(network):

    # note: I replaced first image in alpha, beta, and gamma with my own greek image letters
    (greek_loader, example_data, example_targets,) = createGreekSymbolSet(
        "csvSymbolDataMyExamples.csv",
        "csvSymbolCategoriesMyExamples.csv",
        "greekWithMyExamples",
    )

    trunc_network = createTruncModel(example_data, example_targets)
    totalArr = projectGreekSymbols(trunc_network, "csvSymbolDataMyExamples.csv")
    computeDistEmbeddingSpace(trunc_network, totalArr, "csvSymbolDataMyExamples.csv")

    return


# main calls functions to create a digit embedding space
def main(argv):

    # read in the network
    network_model = torch.load("networkSaved.pt")
    network_model.eval()

    # Part 3A
    (
        greek_loader,
        example_data,
        example_targets,
    ) = createGreekSymbolSet("csvSymbolData.csv", "csvSymbolCategories.csv", "greek")

    # Part 3B
    trunc_network = createTruncModel(example_data, example_targets)

    # Part 3C
    totalArr = projectGreekSymbols(trunc_network, "csvSymbolData.csv")

    # Part 3D
    computeDistEmbeddingSpace(trunc_network, totalArr, "csvSymbolData.csv")

    # Part 3E
    # took my own images and replaced first images within the alpha, beta, gamma the directories
    ownGreekSymbolData(network_model)

    return


if __name__ == "__main__":
    main(sys.argv)
