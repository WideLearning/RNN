import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from data import xor_loaders
from model import LSTM, RNN
from trainer import validate

torch.random.manual_seed(1)

dl_train, dl_val = xor_loaders(
    seq_count=10**4,
    seq_len=8,
    batch_size=1
)

net = LSTM(
    x_size=2,
    h_size=16,
    y_size=2,
    config={
        "f_hh": "tanh",
        "f_hy": "logsoftmax",
    },
)
net.load_state_dict(torch.load("opt_lstm.p"))
x, y = next(iter(dl_val))
x.requires_grad_(True)
print("shapes", x.shape, y.shape)

batch_size, seq_len = y.shape
state = net.init_state(std=0, batch_size=batch_size)
loss_fn = nn.NLLLoss(reduction="sum")

loss = 0
for pos in range(seq_len):
    p, state = net(x[:, pos], state)
    loss += loss_fn(p, y[:, pos])
loss /= batch_size
print(loss.item())
loss.backward()
for pos in range(seq_len):
    print(f"x[{pos}] gradient:", x.grad[:, pos].abs().sum().item())