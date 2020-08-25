import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-7):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        # Expect x to be N, T, D
        assert x.shape[-1] == self.dim
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return ((x - mean) / std * self.gamma) + self.beta


class AddNorm(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.layernorm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, Fx):
        return self.layernorm(x + self.dropout(Fx))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.addnorm = AddNorm(d_model, dropout)

    def forward(self, x0):
        x1 = F.relu(self.fc1(x0))
        Fx = self.fc2(x1)
        return self.addnorm(x0, Fx)


class MultiHeadAttention(nn.Module):
    pass
