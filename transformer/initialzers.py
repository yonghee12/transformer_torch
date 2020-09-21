from torch.nn.init import *


def xavier(*shape, requires_grad=True):
    w = torch.empty(*shape, requires_grad=requires_grad)
    return xavier_normal_(w)


def he(*shape, requires_grad=True):
    w = torch.empty(*shape, requires_grad=requires_grad)
    return kaiming_normal_(w, mode='fan_out', nonlinearity='relu')


def randint10(*shape, requires_grad=False):
    return torch.randint(low=0, high=10, size=(*shape,), requires_grad=requires_grad)


def randint100(*shape, requires_grad=False):
    return torch.randint(low=0, high=100, size=(*shape,), requires_grad=requires_grad)
