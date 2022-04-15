# Kaelyn Jefferson
# Project 5: Recognition using Deep Networks
# Question 3: digit embedding space
# BUILD WITH: python3 digitEmbeddingSpace.py

# import statements
from ctypes import sizeof
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
from os import listdir
from PIL import Image
import csv
from subModel import secondSubModel


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

        # flattening operation to 1D vec
        x = x.view(-1, 320)
        """

        # relu on the output?
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        # log.softmax function applied to the output
        return F.log_softmax(x)
        """
        return x


def createGreekSymbolSet():

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
        reg_image = cv2.imread(totalPath, 0)
        current_img = np.invert(reg_image)
        cv2.imwrite(totalPath, current_img)

        # convert nested array to one arr to save to csv
        # https://thispointer.com/python-convert-list-of-lists-or-nested-list-to-flat-list/
        arrOfValues = []
        for each in current_img:
            arrOfValues.extend(each)
            print(len(each))
        print(len(arrOfValues))

        # writes to csv with values in
        with open(
            "csvSymbolData.csv",
            "a",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(arrOfValues)

        # img = cv2.imread(totalPath)
        # print(len(img))
        # TODO: why is this returning 133? size is 28x28
        # converting between cvmat and other

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

    return


def createTruncModel(network_model):

    # read in the trained model (from main)
    # build a new model from the old network that terminates at the Dense layer with 50 outputs
    # create object
    new_model = secondSubModel()

    # load state dictionary
    new_model.load_state_dict(
        torch.load(
            "/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/cs5300-project5-RecognitionUsingDeepNetworks/network.pt"
        )
    )
    new_model.eval()

    with torch.no_grad():

        fig = plt.figure()
        for i in range(10):

            firstImage = cv2.imread("data/numsjpg/image0.jpeg")

            # convert tensor to numpy for filter2d fxn : https://www.codegrepper.com/code-examples/python/convert+a+tensor+to+numpy+array
            # sourced code on docs: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
            kernel = new_model.conv1.weight[i, 0].numpy()
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


def projectGreekSymbols(network):

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
    model = Submodel()

    # Part 3A
    # createGreekSymbolSet() #prob: doesnt return correct num of intensity values?

    # Part 3B
    # trunc_network = createTruncModel(network_model) #prob: doesn't return b/c can't find fc2

    # Part 3C
    # projectGreekSymbols(trunc_network)

    # Part 3D
    # computeDistEmbeddingSpace()

    # Part 3E
    # ownGreekSymbolData() #basically similar to part D using these eamples

    return


if __name__ == "__main__":
    main(sys.argv)
