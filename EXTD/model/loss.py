import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(input, target):
    return F.cross_entropy(input, target)


def smooth_l1_loss(input, target):
    return F.smooth_l1_loss(input, target)
