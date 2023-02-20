import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from data import xor_loaders
from rnn import RNN, LSTM
from transformer import LinearAttention, SoftmaxAttention, SumParallel
from teleboard.tracker import ConsoleTracker, FileTracker
from trainer import RNNTrainer, FeedforwardTrainer

# net = LSTM(
#     x_size=2,
#     h_size=16,
#     y_size=2,
#     config={
#         "f_hh": "tanh",
#         "f_hy": "logsoftmax",
#     },
# )

net = nn.Sequential(
    SumParallel(nn.Linear(2, 16), SoftmaxAttention(2, 16, 16)),
    SumParallel(nn.Linear(16, 2), SoftmaxAttention(16, 16, 2)),
    nn.LogSoftmax(dim=-1),
)

for l in range(5, 100):
    dl_train, dl_val = xor_loaders(
        seq_count=10**4,
        seq_len=l,
        batch_size=128,
    )

    trainer = FeedforwardTrainer(
        net=net,
        make_opt=lambda p: torch.optim.Adam(p),
        dl_train=dl_train,
        dl_val=dl_val,
        tracker=ConsoleTracker(k=5, regex=".*.loss"),
        # tracker=FileTracker(k=5, filename="rnn.p"),
        train_batches=100,
        val_batches=3,
    )

    trainer.train(n_epochs=1000)
