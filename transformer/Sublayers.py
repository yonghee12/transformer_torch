from .Modules import *


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

    def forward(self, x0):
        x1 = F.relu(self.fc1(x0))
        Fx = self.fc2(x1)
        return Fx


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads if d_k is None else d_k
        self.d_model = d_model
        self.n_heads = n_heads

        self.Wq = nn.Parameter(he(d_model, self.d_k * n_heads), requires_grad=True)
        self.Wk = nn.Parameter(he(d_model, self.d_k * n_heads), requires_grad=True)
        self.Wv = nn.Parameter(he(d_model, self.d_k * n_heads), requires_grad=True)
        self.Wo = nn.Parameter(he(self.d_k * n_heads, d_model), requires_grad=True)

        # method 1 사용시
        self.attentions = nn.ModuleList([ScaledDotProductAttention(self.d_k, dropout) for _ in range(n_heads)])
        # method 2 사용시
        self.attention = ScaledDotProductAttention(self.d_k, dropout)

    def forward(self, q, k, v, mask=None):
        # N: n_batch, T: seq_len
        # q, k, v는 (N, T, d_model), Wq, Wk, Wv는 (d_model, d_k)
        # Q, K, V는 N, T, d_k * n_heads (논문대로 학습하면 d_k * n_heads는 d_model과 같도록)

        Q = q @ self.Wq
        K = k @ self.Wk
        V = v @ self.Wv

        n_batch, q_seq_len, k_seq_len, n_heads, d_k = Q.shape[0], Q.shape[1], K.shape[1], self.n_heads, self.d_k

        # 1. 논문 그대로 잘라서 각각 attention한 후 합치는 방법
        # (N, T, d_k * n_heads)를 n_heads로 chunk 나누어 하나의 head: (N, T, d_k)를 만듦

        # q_heads = torch.chunk(Q, chunks=self.n_heads, dim=2)                            # n_heads, N, T, d_k
        # k_heads = torch.chunk(K, chunks=self.n_heads, dim=2)                            # n_heads, N, T, d_k
        # v_heads = torch.chunk(V, chunks=self.n_heads, dim=2)                            # n_heads, N, T, d_k
        # heads_output = [attention(qh, kh, vh) for qh, kh, vh, attention in              # n_heads, N, T, d_k
        #                 zip(q_heads, k_heads, v_heads, self.attentions)]
        # heads_concat = torch.cat(heads_output, dim=2)                                   # N, T, d_model
        # output = heads_concat

        # 2. Vectorized Method
        # 목적: N, n_heads, T, d_k 모양을 만들어서 attention 진행
        Q_ = torch.transpose(Q.view(n_batch, q_seq_len, self.n_heads, self.d_k), 1, 2)   # N, n_heads, T, d_k
        K_ = torch.transpose(K.view(n_batch, k_seq_len, self.n_heads, self.d_k), 1, 2)   # N, n_heads, T, d_k
        V_ = torch.transpose(V.view(n_batch, k_seq_len, self.n_heads, self.d_k), 1, 2)   # N, n_heads, T, d_k
        output = self.attention(Q_, K_, V_, mask)                                      # N, n_heads, T, d_k
        output = output.transpose(1, 2)                                                # N, T, n_heads, d_k
        output = output.contiguous().view(n_batch, q_seq_len, self.n_heads * self.d_k)   # N, T, n_heads * self.d_k(=d_model)

        return output @ self.Wo
