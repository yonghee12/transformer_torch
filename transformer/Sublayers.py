from math import sqrt

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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.sqrt_d_k = sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        K_t = K.transpose(1, 2)
        scaled_dot = (Q @ K_t) / self.sqrt_d_k  # (N, T, d_k) * (N, T, d_k) -> (N, T, T)
        attention_score = F.softmax(scaled_dot, dim=2)  # N, T, T의 dimension 중 마지막 dimension 기준으로 softmax
        return attention_score @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_heads if d_k is None else d_k
        self.attentions = nn.Module([ScaledDotProductAttention(self.d_k, dropout) for _ in range(n_heads)])

    def forward(self, q, k, v):
        Q = q @ Wq
        K = k @ Wk
        V = v @ Wv

        multiheads = np.hstack((head1, head2, head3, head4, head5, head6, head7, head8))  # shape: (s, 64 * 8)
        output = multiheads @ W  # shape: (s, 512) @ (512, 512) -> (s, 512)
