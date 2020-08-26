from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .Layers import *


class EncoderBlock(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock:
    def __init__(self, n_layers, d_model, d_ff, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class InputBlock(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)


class OutputBlock(nn.Module):
    def __init__(self):
        super().__init__()
