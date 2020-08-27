from .Sublayers import *


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.addnorms = nn.ModuleList([AddNorm(d_model, dropout=dropout) for _ in range(2)])

    def forward(self, x, enc_mask):
        x1 = self.addnorms[0](x, self.self_attention(q=x, k=x, v=x, mask=enc_mask))
        x2 = self.addnorms[1](x1, self.feed_forward(x1))
        return x2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.encoder_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.addnorms = nn.ModuleList([AddNorm(d_model, dropout=dropout) for _ in range(3)])

    def forward(self, x, enc_out, enc_mask, dec_mask):
        x1 = self.addnorms[0](x, self.self_attention(q=x, k=x, v=x, mask=dec_mask))
        x2 = self.addnorms[1](x1, self.encoder_attention(q=x1, k=enc_out, v=enc_out, mask=enc_mask))
        x3 = self.addnorms[2](x2, self.feed_forward(x2))
        return x3

