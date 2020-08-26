from .Sublayers import *


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x):
        x1 = self.self_attention(x)
        x2 = self.feed_forward(x1)
        return x2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.encoder_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, enc_out):
        x1 = self.self_attention(x, x, x)
        x2 = self.encoder_attention(x1, enc_out, enc_out)
        x3 = self.feed_forward(x2)
        return x3


class InputLayer:
    pass


class OutputLayer:
    pass
