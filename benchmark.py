import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from data import xor_loaders
from rnn import LSTM, RNN
from teleboard.tracker import ConsoleTracker, FileTracker
from trainer import TeacherForcingTrainer

dl_train, dl_val = xor_loaders(
    seq_count=10**4,
    seq_len=3,
    batch_size=128,
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

trainer = TeacherForcingTrainer(
    net=net,
    make_opt=lambda p: torch.optim.Adam(p),
    dl_train=dl_train,
    dl_val=dl_val,
    tracker=FileTracker(k=5, filename="rnn.p"),
    train_batches=100,
    val_batches=3,
)

torch.save(net.state_dict(), "rnd_rnn.p")

trainer.train(n_epochs=100)

torch.save(net.state_dict(), "opt_rnn.p")
