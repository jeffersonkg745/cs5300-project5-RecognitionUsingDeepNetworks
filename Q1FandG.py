# Kaelyn Jefferson
# Project 5: Recognition using Deep Networks -- Question 1F and 1G


# import statements
import torch
import torchvision
import buildAndTrainNetwork
import sys
import matplotlib.pyplot as plt
from buildAndTrainNetwork import MyNetwork
import numpy as np
import os
import cv2


# Source: Dr. Maxwell example approach
class CustomTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        # x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.resize(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# read the network and run it on the test set
def runOnTestSet(network_model, example_data, example_targets):

    with torch.no_grad():
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


# test the network on new inputs (use my own digits from 0 to 9)
# folder structure shown here: https://stackoverflow.com/questions/49073799/pytorch-testing-with-torchvision-datasets-imagefolder-and-dataloader
# code sourced from: https://www.kaggle.com/code/leifuer/intro-to-pytorch-loading-image-data/notebook
# code sourced from: https://pytorch.org/vision/0.8/datasets.html#imagefolder
def prepare_custom_data_set():

    # Source: Dr. Maxwell example approach
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            "custom_nums",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    CustomTransform(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=27,
        shuffle=False,
    )

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


# calls the functions to read the network and test on the mnist test set and new input set
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
