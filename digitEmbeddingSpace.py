# Kaelyn Jefferson
# Project 5: Recognition using Deep Networks
# Question 3: digit embedding space
# BUILD WITH: python3 digitEmbeddingSpace.py

# import statements
from ctypes import sizeof
from tkinter.dnd import dnd_start
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


# class definitions
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

        # relu on the output?
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # x = self.fc2(x)

        # log.softmax function applied to the output
        return F.log_softmax(x)
        # return x


# source: Maxwell
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# source: Maxwell
def createGreekSymbolSet():

    greek_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            "greek",
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

    # use same code from MNIST for input and applying to the network

    """
    # resets the csv file when you want to create a new greek symbol set
    with open(
        "csvSymbolData.csv",
        "w",
    ) as f:
        writer = csv.writer(f)
        writer.writerow("Data for Greek Symbol data set:")

    # resets the csv file when you want to create a new greek symbol set
    with open(
        "csvSymbolCategories.csv",
        "w",
    ) as f:
        writer = csv.writer(f)
        writer.writerow("Categories for Greek Symbol data set:")

    # referenced code here: https://www.geeksforgeeks.org/how-to-iterate-through-images-in-a-folder-python/
    # https://www.tutorialspoint.com/pytorch-how-to-resize-an-image-to-a-given-size
    # https://www.tutorialspoint.com/pytorch-how-to-convert-an-image-to-grayscale
    # https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
    # https://www.delftstack.com/howto/python/opencv-invert-image/
    folder_directory = "data/greek-1"
    for images in os.listdir(folder_directory):
        path = "/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/cs5300-project5-RecognitionUsingDeepNetworks/data/greek-1/"
        totalPath = path + images
        print(totalPath)
        current_img = Image.open(totalPath)

        # resize to 28x28
        resized_img = torchvision.transforms.Resize((28, 28))
        current_img = resized_img(current_img)

        # convert to grayscale
        toGray = torchvision.transforms.Grayscale()
        current_img = toGray(current_img)

        # invert image intensities
        current_img = torchvision.transforms.functional.invert(current_img)
        print(np.asarray(current_img))  # PRINTED HERE
        current_img = np.asarray(current_img)
        intensity_values = []
        for row in current_img:
            for each in row:
                # print(each)
                intensity_values.append(each)

        # writes to csv with values in
        with open(
            "csvSymbolData.csv",
            "a",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(intensity_values)

        # put values in a dictionary here
        categoryDict = {"alpha": 0, "beta": 1, "gamma": 2}
        subStringOfName = images.split("_")[0]
        if subStringOfName not in categoryDict:
            categoryDict[subStringOfName] = len(categoryDict)

        # writes each line's symbol categories
        with open(
            "csvSymbolCategories.csv",
            "a",
            newline="",
        ) as f:
            writer = csv.writer(f)
            num = categoryDict.get(subStringOfName)
            writer.writerow(str(num))

    """

    return


def createTruncModel():

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
    print(new_model)  # shows output features is 50

    # apply network to first training image and see 50 output numbers

    firstImage = cv2.imread("data/custom_nums_all/IMG_0280.jpeg")

    with torch.no_grad():

        fig = plt.figure()
        for i in range(10):

            # convert tensor to numpy for filter2d fxn : https://www.codegrepper.com/code-examples/python/convert+a+tensor+to+numpy+array
            # sourced code on docs: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
            kernel = new_model.conv1.weight[i, 0].numpy()
            dst = cv2.filter2D(firstImage, -1, kernel)

            print(len(dst[0]))

    return new_model


def projectGreekSymbols(trunc_network):

    # read from the csv file
    # PROB: convert intensity --> image
    """
    with open(
        "csvSymbolData.csv",
        newline="",
    ) as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            # print(row)
            # print("a new row is here")
            # convert back to image space with 28x28 values

            a = np.arange(784).reshape((28, 28))
            print(a)

            current_image = Image.fromarray(np.uint8(a)).convert("L")
            current_image.save("ConvertedBackPhoto.png")

            print(current_image)
        # writer = csv.writer(f)
        # writer.writerow(intensity_values)
    """

    # read from image folder for now + apply truncated network to get a set of 27 50-element vecs
    fig = plt.figure()
    i = 0

    folder_directory = "data/greek-1"
    for images in os.listdir(folder_directory):
        path = "/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/cs5300-project5-RecognitionUsingDeepNetworks/data/greek-1/"
        totalPath = path + images
        print(totalPath)
        # current_img = Image.open(totalPath)  # current image object
        current_img = cv2.imread(totalPath)

        # similar as above
        with torch.no_grad():
            for i in range(9):
                kernel = trunc_network.conv1.weight[i, 0].numpy()
                dst = cv2.filter2D(current_img, -1, kernel)
                # print(dst)

            plt.subplot(3, 3, i + 1)
            plt.tight_layout()
            print(dst)
            plt.imshow(dst, cmap="gray", interpolation="none")
            plt.title("Img #: {}".format(i))
            plt.xticks([])
            plt.yticks([])
            i += 1
    print(fig)
    plt.show()

    return


def computeDistEmbeddingSpace(network):

    return


def ownGreekSymbolData(network):

    return


# main function (yes, it needs a comment too)
def main(argv):

    # read in the network
    network_model = torch.load("networkSaved.pt")
    network_model.eval()

    # make a submodel that includes everything but the output layer
    # model = Submodel()

    # Part 3A
    createGreekSymbolSet()

    # Part 3B
    # trunc_network = createTruncModel()

    # Part 3C
    # projectGreekSymbols(trunc_network)

    # Part 3D
    # computeDistEmbeddingSpace()

    # Part 3E
    # ownGreekSymbolData() #basically similar to part D using these eamples

    return


if __name__ == "__main__":
    main(sys.argv)
