import torch
from torch import nn
from torch.nn import functional as F

from .blocks import *


class Transformer(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, max_seq_len_enc, max_seq_len_dec, pad_idx=0, d_model=512, d_ff=2048, n_layers=8, n_heads=8,
                 activation='relu', dropout=0.1, dec_emb_output_weight_share=False, enc_dec_emb_weight_share=False):
        super().__init__()
        self.enc_input = TransformerInputBlock(enc_vocab_size, d_model, max_seq_len_enc, pad_idx, dropout=dropout)
        self.dec_input = TransformerInputBlock(enc_vocab_size, d_model, max_seq_len_dec, pad_idx, dropout=dropout)
        self.encoder = TransformerEncoderBlock(n_layers, d_model, d_ff, n_heads, activation, dropout)
        self.decoder = TransformerDecoderBlock(n_layers, d_model, d_ff, n_heads, activation, dropout)
        self.output = TransformerOutputBlock(d_model, dec_vocab_size)

        if enc_dec_emb_weight_share:
            self.enc_input.embedding.weight = self.dec_input.embedding.weight

        if dec_emb_output_weight_share:
            raise NotImplementedError

    def forward(self, enc_input, dec_input):
        """
        :param enc_input: masked 2d tensor (batch_size, seq_len)
        :param dec_input: masked 2d tensor (batch_size, seq_len)
        """
        enc_mask, dec_mask = get_padding_mask(enc_input), get_padding_mask(dec_input)

        enc_emb, dec_emb = self.enc_input(enc_input), self.dec_input(dec_input)
        enc_output = self.encoder(enc_emb, enc_mask)
        dec_output = self.decoder(dec_emb, enc_output, enc_mask, dec_mask)

        return self.output(dec_output)
