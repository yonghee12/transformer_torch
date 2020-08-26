from .Sublayers import *


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()

    def forward(self, enc_input):
        x1 = self.self_attention(enc_input)
        x2 = self.feed_forward(x1)
        return x2


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.encoder_attention = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()

    def forward(self, x0, enc_out, ):
        x1 = self.self_attention(x0, x0, x0)
        x2 = self.encoder_attention(x1, enc_out, enc_out)
        x3 = self.feed_forward(x2)
        return x3


class InputLayer:
    pass


class OutputLayer:
    pass
