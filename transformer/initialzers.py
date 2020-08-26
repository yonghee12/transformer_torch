import torch
from torch.nn.init import *


def xavier(*shape, requires_grad=False):
    w = torch.empty(*shape, requires_grad=requires_grad)
    return xavier_normal_(w)


def he(*shape, requires_grad=True):
    w = torch.empty(*shape, requires_grad=requires_grad)
    return kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
