from torch.optim import Adam

from transformer.Initialzers import *
from transformer.Models import *

device = torch.device('cuda:0')

enc_vocab = 11
dec_vocab = 11
d_model = 8
d_ff = 16
n_layers = 1
n_heads = 2
dropout = 0.0

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