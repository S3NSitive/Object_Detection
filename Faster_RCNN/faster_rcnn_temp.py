import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Choose Device for Torch
# device='cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = (800, 800)


