from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .Layers import *


class EncoderBlock(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x, enc_mask):
        for layer in self.layers:
            x = layer(x, enc_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x, enc_output, enc_mask, dec_mask):
        for layer in self.layers:
            x = layer(x, enc_output, enc_mask, dec_mask)
        return x


class InputBlock(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_len, pad_idx, dropout=0.1):
        # to=0: Encoder, to=1: Decoder
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positional_encodeing = PositionalEncoding(seq_len, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: 2d tensor (batch_size, seq_len)
        :return: 3d tensor (batch_size, seq_len, embedding_dim)
        """
        x_emb = self.embedding(x)
        x_pos = self.positional_encodeing(x_emb)
        return self.dropout(x_pos)



class OutputBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass