# Question 1F: read network + run on test set

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

# read the network
model = torch.load("networkSaved.pt")
model.eval()
