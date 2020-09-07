import pandas as pd
import numpy as np

from torch.optim import Adam

from transformer.Initialzers import *
from transformer.Models import *

from langframe import *
from train.functions import *

device = torch.device('cuda:0')
tokenizer_kor = KoreanTokenizer('mecab')

df = pd.read_excel('data/korean-english-parallel/2_대화체_200226.xlsx')
df = df[df['대분류'] == '일상대화']
kor, eng = df['원문'].tolist(), df['번역문'].tolist()

tokenized_matrix_eng = get_tokenized_matrix(eng, 'word_tokenize', False, [])
tokenized_matrix_kor = [tokenizer_kor.morphs(text) for text in kor]

# add <sos>, <eos> tokens
tokenized_matrix_eng = [["<sos>"] + tokens + ["<eos>"] for tokens in tokenized_matrix_eng]
tokenized_matrix_kor = [["<sos>"] + tokens + ["<eos>"] for tokens in tokenized_matrix_kor]

# token2idx, idx2token 생성
unique_tokens_eng, unique_tokens_kor = get_uniques_from_nested_lists(tokenized_matrix_eng), get_uniques_from_nested_lists(tokenized_matrix_kor)
token2idx_eng, idx2token_eng = get_item2idx(unique_tokens_eng, unique=True, start_from_one=True)
token2idx_kor, idx2token_kor = get_item2idx(unique_tokens_kor, unique=True, start_from_one=True)

# max_seq_len
max_seq_len_enc = get_max_seq_len(tokenized_matrix_kor)
max_seq_len_dec = get_max_seq_len(tokenized_matrix_eng)

# making sequence-wise inputs
pass

# padding
padded_enc = pad_sequence_nested_lists(tokenized_matrix_kor, max_len=max_seq_len_enc, method='post', truncating='post')
padded_dec = pad_sequence_nested_lists(tokenized_matrix_eng, max_len=max_seq_len_dec, method='post', truncating='post')
X_enc = [[token2idx_kor[token] for token in row] for row in padded_enc]
X_dec = [[token2idx_eng[token] for token in row] for row in padded_dec]

sample = tokenized_matrix_eng[0]
sample

enc_vocab = len(unique_tokens_kor)
dec_vocab = len(unique_tokens_eng)
d_model = 32
d_ff = 128
n_layers = 3
n_heads = 2
dropout = 0.05

x_enc = randint10(20, 5).to(device)
x_dec = randint10(20, 4).to(device)
y_true = randint10(20, 1).to(device)

model = Transformer(enc_vocab, dec_vocab, x_enc.shape[1], -1, d_model, d_ff, n_layers, n_heads, dropout)
model = model.to(device)
optimizer = Adam(model.parameters())

n_epochs = 1000
epoch_loss = 0
for epoch in range(n_epochs):
    log_y_pred = model(x_enc, x_dec)
    log_y_pred = log_y_pred[:, -1, :]
    loss = F.nll_loss(input=log_y_pred, target=y_true.squeeze(), reduction='mean')

    optimizer.zero_grad()
    loss.backward()
    epoch_loss = loss.item()
    optimizer.step()

    print(f"after epoch: {epoch}, epoch_losses: {round(epoch_loss, 3)}")
