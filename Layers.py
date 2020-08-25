from Sublayers import *

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()

    def forward(self, x0):
        x1 = self.multi_head_attention(x0)
        x2 = self.feed_forward(x1)


class DecoderLayer:
    pass

class InputLayer:
    pass
