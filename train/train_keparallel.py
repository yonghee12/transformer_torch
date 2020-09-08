from time import perf_counter

from direct_redis import DirectRedis
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from langframe import *
from train.functions import *
from transformer.Models import *

REDIS = True

r = DirectRedis(host='127.0.0.1', port='6379')
device = torch.device('cuda:0')
tokenizer_kor = KoreanTokenizer('mecab')


def get_corpus_preprocessed():
    df = pd.read_excel('data/korean-english-parallel/2_대화체_200226.xlsx')
    df = df[df['대분류'] == '일상대화']
    kor, eng = df['원문'].tolist(), df['번역문'].tolist()

    tokenized_matrix_eng = get_tokenized_matrix(eng, 'word_tokenize', False, [])
    tokenized_matrix_kor = [tokenizer_kor.morphs(text) for text in kor]

    # add <sos>, <eos> tokens
    tokenized_matrix_eng = [["<sos>"] + tokens + ["<eos>"] for tokens in tokenized_matrix_eng]
    tokenized_matrix_kor = [["<sos>"] + tokens + ["<eos>"] for tokens in tokenized_matrix_kor]

    # token2idx, idx2token 생성
    unique_tokens_eng, unique_tokens_kor = get_uniques_from_nested_lists(
        tokenized_matrix_eng), get_uniques_from_nested_lists(tokenized_matrix_kor)
    token2idx_eng, idx2token_eng = get_item2idx(unique_tokens_eng, unique=True, start_from_one=True)
    token2idx_kor, idx2token_kor = get_item2idx(unique_tokens_kor, unique=True, start_from_one=True)

    # max_seq_len
    max_seq_len_enc = get_max_seq_len(tokenized_matrix_kor)
    max_seq_len_dec = get_max_seq_len(tokenized_matrix_eng)

    # making sequence-wise inputs
    X_enc, X_dec, y_true = [], [], []
    for idx, tokens in enumerate(tokenized_matrix_eng):
        X_enc.extend([tokenized_matrix_kor[idx] for _ in range(len(tokens) - 1)])
        X_dec.extend([tokens[:seq_i] for seq_i in range(1, len(tokens))])
        y_true.extend(tokens[1:])

    # padding
    padded_enc = pad_sequence_nested_lists(X_enc, max_len=max_seq_len_enc, method='post', truncating='post')
    padded_dec = pad_sequence_nested_lists(X_dec, max_len=max_seq_len_dec, method='post', truncating='post')
    X_enc = [[token2idx_kor[token] for token in row] for row in padded_enc]
    X_dec = [[token2idx_eng[token] for token in row] for row in padded_dec]
    y_true = [token2idx_eng[token] for token in y_true]
    corpus_set = [X_enc, X_dec, y_true, unique_tokens_kor, unique_tokens_eng, max_seq_len_enc, max_seq_len_dec]

    res = r.hset('keparallel', 'ilsang', corpus_set)
    print(f"Set redis reponse: {res}")
    return corpus_set


X_enc, X_dec, y_true, unique_tokens_kor, unique_tokens_eng, max_seq_len_enc, max_seq_len_dec = \
    r.hget('keparallel', 'ilsang') if REDIS else get_corpus_preprocessed()

X_enc_ = torch.tensor(X_enc, device=device, requires_grad=False)
X_dec_ = torch.tensor(X_dec, device=device, requires_grad=False)
y_true_ = torch.tensor(y_true, device=device, requires_grad=False)

enc_vocab = len(unique_tokens_kor) + 1
dec_vocab = len(unique_tokens_eng) + 1
d_model = 32
d_ff = 128
n_layers = 3
n_heads = 2
dropout = 0.1
# batch_size = len(X_enc) // 100
batch_size = 1500

train_ds = TensorDataset(X_enc_, X_dec_, y_true_)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

model = Transformer(enc_vocab, dec_vocab, max_seq_len_enc, max_seq_len_dec, 0, d_model, d_ff, n_layers, n_heads,
                    dropout, False, False)
model = model.to(device)
optimizer = Adam(model.parameters())

# total: 120
n_epochs = 120
print_all = True
verbose = 1
progresses = {int(n_epochs // (100 / i)): i for i in range(1, 101, 1)}
t0, durations = perf_counter(), list()

model.train()
for epoch in range(n_epochs):
    epoch_loss = 0
    for iteration, ds in enumerate(train_dl):
        X_encoder, X_decoder, y_true_dec = ds
        log_y_pred = model(X_encoder, X_decoder)
        log_y_pred = log_y_pred[:, -1, :]
        loss = F.nll_loss(input=log_y_pred, target=y_true_dec.squeeze(), reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

        if verbose >= 2:
            loss_s = round(loss.item(), 3)
            perp = round(np.exp(loss_s).item(), 2)
            print(f"epoch-iter: {epoch}-{iteration}, loss: {loss_s}, perp: {perp}")

    durations.append(perf_counter() - t0)
    t0 = perf_counter()
    if print_all or epoch in progresses:
        loss_s = round(epoch_loss / (iteration + 1), 3)
        perp = round(np.exp(loss_s).item(), 2)
        print(f"epoch: {epoch}, loss: {loss_s}, perp: {perp}")

if verbose > 0:
    avg_epoch_time = sum(durations) / len(durations)
    print("average epoch time:", round(avg_epoch_time, 3))

# Generator

token2idx_eng, idx2token_eng = get_item2idx(unique_tokens_eng, unique=True, start_from_one=True)
token2idx_kor, idx2token_kor = get_item2idx(unique_tokens_kor, unique=True, start_from_one=True)

with open("data/korean-english-parallel/test.txt") as f:
    testdata = f.readlines()

for line in testdata:
    # input_s = "우리 내일 어디로 갈까?"
    input_s = line
    input_tokens = tokenizer_kor.morphs(input_s.strip())
    input_tokens = ["<sos>"] + input_tokens + ["<eos>"]
    input_padded = pad_sequence_list(input_tokens, max_len=max_seq_len_enc, method='post', truncating='post')
    try:
        X_input = [token2idx_kor[token] for token in input_padded]
    except KeyError as e:
        print(str(e))
        # continue

    gen = ['<sos>']
    for _ in range(max_seq_len_dec):
        gen_padded = pad_sequence_list(gen, max_len=max_seq_len_dec, method='post', truncating='post')
        X_gen = [token2idx_eng[token] for token in gen_padded]

        X_input_ = torch.tensor(X_input, device=device, requires_grad=False).unsqueeze(0)
        X_gen_ = torch.tensor(X_gen, device=device, requires_grad=False).unsqueeze(0)

        model.eval()
        log_y_pred = model(X_input_, X_gen_).squeeze()
        log_y_pred = log_y_pred[-1, :]
        next_gen = torch.argmax(log_y_pred).item()
        next_gen_s = idx2token_eng[next_gen]
        gen.append(next_gen_s)
        # print(' '.join(gen))
        if next_gen_s == '<eos>':
            print(input_s.strip())
            print(' '.join(gen))
            break


log_y_pred = model(X_encoder, X_decoder)
log_y_pred = log_y_pred[:, -1, :]
torch.argmax(log_y_pred)
# vocabs = set()
# for row in X_enc:
#     vocabs.update(set(row))
# len(vocabs)
# len(unique_tokens_kor)

torch.save()