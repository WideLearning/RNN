import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from data import xor_loaders
from model import LSTM, RNN
from trainer import validate

dl_train, dl_val = xor_loaders(
    seq_count=10**4,
    seq_len=5,
    batch_size=128,
    allow_intermediate=False
)

net = RNN(
    x_size=2,
    h_size=8,
    y_size=2,
    config={
        "f_hh": "tanh",
        "f_hy": "logsoftmax",
    },
)

rnd_dict = torch.load("rnd_rnn.p")
opt_dict = torch.load("opt_rnn.p")


def test(alpha):
    cur_dict = {x: (1 - alpha) * rnd_dict[x] + alpha * opt_dict[x] for x in rnd_dict}
    net.load_state_dict(cur_dict)
    losses = validate(net, dl_val, 100)
    return sum(losses) / len(losses)


x = np.linspace(0, 2, 100)
y = [test(t) for t in x]
plt.plot(x, y, 'o')
plt.show()