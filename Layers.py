from . import *

class EncoderLayer:
    def __init__(self):
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()

    def forward(self, x0):
        x1 = self.multi_head_attention(x0)
        x2 = self.feed_forward(x1)


class DecoderLayer:
    pass

class InputLayer:
    pass
