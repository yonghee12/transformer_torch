import os
from time import perf_counter

from direct_redis import DirectRedis
from torch.optim import Adam, AdamW, Adagrad, SGD
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from langframe import *
from train.functions import *
from train.preprocess import *
from transformer.Models import *

optimizers = {'Adam': Adam, "AdamW": AdamW, "Adagrad": Adagrad, "SGD": SGD}

REDIS = True
MAKE_MODEL = True
OPTIMIZER = 'AdamW'

r = DirectRedis(host='127.0.0.1', port='6379')
device = torch.device('cuda:0')
tokenizer_kor = KoreanTokenizer('mecab')


def save_checkpoint(directory, model, optimizer, epoch, loss_s, perp):
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint = {
        'model': model,
        'optimizer': optimizer,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    modelpath = os.path.join(directory, f"epoch_{epoch}_loss_{loss_s}_perp_{perp}.checkpoint")
    torch.save(checkpoint, modelpath)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model_ = checkpoint['model']
    model_.load_state_dict(checkpoint['model_state_dict'])
    optimizer_ = checkpoint['optimizer']
    optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])
    return model_, optimizer_


X_enc, X_dec, y_true, unique_tokens_kor, unique_tokens_eng, max_seq_len_enc, max_seq_len_dec = \
    r.hget('keparallel', 'ilsang') if REDIS else get_corpus_preprocessed()

X_enc_ = torch.tensor(X_enc, device=device, requires_grad=False)
X_dec_ = torch.tensor(X_dec, device=device, requires_grad=False)
y_true_ = torch.tensor(y_true, device=device, requires_grad=False)

batch_size = 1150
train_ds = TensorDataset(X_enc_, X_dec_, y_true_)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

if MAKE_MODEL:
    enc_vocab = len(unique_tokens_kor) + 1
    dec_vocab = len(unique_tokens_eng) + 1
    d_model = 128
    d_ff = 512
    n_layers = 6
    n_heads = 4
    dropout = 0.1
    # batch_size = len(X_enc) // 100

    model = Transformer(enc_vocab, dec_vocab, max_seq_len_enc, max_seq_len_dec, 0, d_model, d_ff, n_layers, n_heads,
                        dropout, False, False)
    model = model.to(device)
    optimizer = Adam(model.parameters())
else:
    path = 'trained_models/epoch_119_loss_2.416_perp_11.2.checkpoint'
    model, optimizer = load_checkpoint(path)

token2idx_eng, idx2token_eng = get_item2idx(unique_tokens_eng, unique=True, start_from_one=True)
token2idx_kor, idx2token_kor = get_item2idx(unique_tokens_kor, unique=True, start_from_one=True)
with open("data/korean-english-parallel/test.txt") as f:
    testdata = f.readlines()

# init once
total_epochs = 0

# train codes from here
n_epochs = 30
print_all = True
verbose = 1
progresses = {int(n_epochs // (100 / i)): i for i in range(1, 101, 1)}
t0, durations = perf_counter(), list()

for epoch in range(n_epochs):
    epoch_loss = 0
    model.train()
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
            print(f"epoch-iter: {total_epochs}-{iteration}, loss: {loss_s}, perp: {perp}")

    durations.append(perf_counter() - t0)
    t0 = perf_counter()
    if print_all or epoch in progresses:
        loss_s = round(epoch_loss / (iteration + 1), 3)
        perp = round(np.exp(loss_s).item(), 2)
        print(f"epoch: {total_epochs}, loss: {loss_s}, perp: {perp}")

        directory = 'trained_models'
        save_checkpoint(directory, model, optimizer, total_epochs, loss_s, perp)

        if perp < 50:
            model.eval()
            for line in testdata:
                # input_s = "우리 내일 어디로 갈까?"
                input_s = line.strip()
                input_tokens = tokenizer_kor.morphs(input_s)
                input_tokens = ["<sos>"] + input_tokens + ["<eos>"]
                input_padded = pad_sequence_list(input_tokens, max_len=max_seq_len_enc, method='post', truncating='post')
                try:
                    X_input = [token2idx_kor[token] for token in input_padded]
                except KeyError:
                    print(f'KeyError: {input_s}')
                else:
                    gen = ['<sos>']
                    for _ in range(max_seq_len_dec):
                        gen_padded = pad_sequence_list(gen, max_len=max_seq_len_dec, method='post', truncating='post')
                        X_gen = [token2idx_eng[token] for token in gen_padded]

                        X_input_ = torch.tensor(X_input, device=device, requires_grad=False).unsqueeze(0)
                        X_gen_ = torch.tensor(X_gen, device=device, requires_grad=False).unsqueeze(0)

                        log_y_pred = model(X_input_, X_gen_).squeeze()
                        log_y_pred = log_y_pred[-1, :]
                        next_gen = torch.argmax(log_y_pred).item()
                        next_gen_s = idx2token_eng[next_gen]
                        gen.append(next_gen_s)

                        if next_gen_s == '<eos>':
                            print('원문: ' + input_s.strip())
                            print('번역: ' + ' '.join([tok for tok in gen if tok not in ('<sos>', '<eos>')]))
                            break
                    print('(X) 원문: ' + input_s.strip())

    total_epochs += 1

if verbose > 0:
    avg_epoch_time = sum(durations) / len(durations)
    print("average epoch time:", round(avg_epoch_time, 3))




print()

# Generator
token2idx_eng, idx2token_eng = get_item2idx(unique_tokens_eng, unique=True, start_from_one=True)
token2idx_kor, idx2token_kor = get_item2idx(unique_tokens_kor, unique=True, start_from_one=True)

with open("data/korean-english-parallel/test.txt") as f:
    testdata = f.readlines()

model.eval()
for line in testdata:
    # input_s = "우리 내일 어디로 갈까?"
    input_s = line.strip()
    input_tokens = tokenizer_kor.morphs(input_s)
    input_tokens = ["<sos>"] + input_tokens + ["<eos>"]
    input_padded = pad_sequence_list(input_tokens, max_len=max_seq_len_enc, method='post', truncating='post')
    try:
        X_input = [token2idx_kor[token] for token in input_padded]
    except KeyError:
        print(f'KeyError: {input_s}')
    else:
        gen = ['<sos>']
        for _ in range(max_seq_len_dec):
            gen_padded = pad_sequence_list(gen, max_len=max_seq_len_dec, method='post', truncating='post')
            X_gen = [token2idx_eng[token] for token in gen_padded]

            X_input_ = torch.tensor(X_input, device=device, requires_grad=False).unsqueeze(0)
            X_gen_ = torch.tensor(X_gen, device=device, requires_grad=False).unsqueeze(0)

            # log_y_pred = model(X_input_, X_gen_).squeeze()
            log_y_pred = model(X_input_, X_gen_).squeeze()
            log_y_pred = log_y_pred[-1, :]
            next_gen = torch.argmax(log_y_pred).item()
            next_gen_s = idx2token_eng[next_gen]
            gen.append(next_gen_s)
            # print(' '.join(gen))
            if next_gen_s == '<eos>':
                print('원문: ' + input_s.strip())
                print('번역: ' + ' '.join([tok for tok in gen if tok not in ('<sos>', '<eos>')]))
                break
