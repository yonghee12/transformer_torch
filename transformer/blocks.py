from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .layers import *


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x, enc_mask):
        for layer in self.layers:
            x = layer(x, enc_mask)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x, enc_output, enc_mask, dec_mask):
        for layer in self.layers:
            x = layer(x, enc_output, enc_mask, dec_mask)
        return x


class TransformerInputBlock(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, seq_len: int, pad_idx=0, dropout=0.1,
                 mul_sqrt_dmodel=False):
        # to=0: Encoder, to=1: Decoder
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positional_encodeing = PositionalEncoding(max_seq_len=seq_len, d_model=embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.sqrt_dmodel = sqrt(embedding_dim)
        self.mul_sqrt_dmodel = mul_sqrt_dmodel

    def forward(self, x):
        """
        :param x: 2d tensor (batch_size, seq_len)
        :return: 3d tensor (batch_size, seq_len, embedding_dim)
        """
        x_emb = self.embedding(x)
        x_emb = x_emb if not self.mul_sqrt_dmodel else x_emb * self.sqrt_dmodel
        x_pos = self.positional_encodeing(x_emb)
        return self.dropout(x_pos)


class TransformerOutputBlock(nn.Module):
    def __init__(self, input_embedding, vocab_size):
        super().__init__()
        self.linear = nn.Linear(input_embedding, vocab_size, bias=True)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)
