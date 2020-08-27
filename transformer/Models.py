import torch
from torch import nn
from torch.nn import functional as F

from .Blocks import *


class Transformer(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, pad_idx=0, d_model=512, d_ff=2048, n_layers=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.enc_input = InputBlock(enc_vocab_size, d_model, pad_idx, to=0)
        self.dec_input = InputBlock(dec_vocab_size, d_model, pad_idx, to=1)
        self.encoder = EncoderBlock(n_layers, d_model, d_ff, n_heads, dropout)
        self.decoder = DecoderBlock(n_layers, d_model, d_ff, n_heads, dropout)
        self.output = OutputBlock()

    def forward(self, enc_input, dec_input):
        """
        :param enc_input: masked 2d tensor (batch_size, seq_len)
        :param dec_input: masked 2d tensor (batch_size, seq_len)
        :return:
        """
        enc_mask = enc_input.unsqueeze(1).unsqueeze(2)   # N, n_heads, T, d_k 를 만족시키기 위해. head에 대해 broadcasting 될 수 있도록
        dec_mask = dec_input.unsqueeze(1).unsqueeze(2)   # N, n_heads, T, d_k 를 만족시키기 위해. head에 대해 broadcasting 될 수 있도록

        enc_emb = self.enc_input(enc_input)
        enc_output = self.encoder(enc_emb, enc_mask)
        dec_emb = self.dec_input(dec_input)
        dec_output = self.decoder(dec_emb, enc_output, enc_mask, dec_mask)
        output = self.output(dec_output)
        return output

