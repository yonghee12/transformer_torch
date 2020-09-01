from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .Initialzers import *


def get_padding_mask(x):
    mask = x.unsqueeze(1).unsqueeze(2)  # N, n_heads, T, d_k 를 만족시키기 위해. head에 대해 broadcasting 될 수 있도록
    return mask


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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.sqrt_d_k = sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask):
        K_t = K.transpose(-1, -2)
        scaled_dot = (Q @ K_t) / self.sqrt_d_k  # (T, d_k) * (T, d_k) -> (T, T)
        scaled_dot = scaled_dot if mask is None else scaled_dot.masked_fill(mask == 0, -1e9)
        attention_score = self.dropout(F.softmax(scaled_dot, dim=-1))  # T, T 중 마지막 dimension 기준으로 softmax
        return attention_score @ V


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pe = self.get_positional_encoding(max_seq_len, d_model)

    def forward(self, x):
        x1 = x + torch.tensor(self.pe[:, :x.size(1)]).clone().detach()
        return x1

    @staticmethod
    def get_positional_encoding(max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (T, 1) 텐서 만들어 broadcasting 유도
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                -math.log(10000.0) / d_model))  # (d_model/2) size 1d 텐서로 broadcasting 유도
        multiplied = position * div_term  # sin, cos 안에 들어가는 term은 같음
        pe[:, 0::2] = torch.sin(multiplied)
        pe[:, 1::2] = torch.cos(multiplied)
        pe = pe.unsqueeze(0)        # 3차원인 embeddin 이후 값에 대응하기 위함
        return pe
