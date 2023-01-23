import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from data import xor_loaders
from model import LSTMA, CECA, LSTM, CEC, RNN
from TeleBoard.tracker import ConsoleTracker
from trainer import TeacherForcingTrainer

dl_train, dl_val = xor_loaders(
    seq_count=10**4,
    seq_len=10,
    batch_size=128,
)

# net = RNN(
#     x_size=2,
#     h_size=4,
#     y_size=2,
#     config={
#         "f_hh": "tanh",
#         "f_hy": "logsoftmax",
#     },
# )

net = CECA(
    x_size=2,
    h_size=64,
    y_size=2,
    config={
        "f_hh": "tanh",
        "f_hy": "logsoftmax"
    },
)

trainer = TeacherForcingTrainer(
    net=net,
    make_opt=lambda p: torch.optim.Adam(p),
    dl_train=dl_train,
    dl_val=dl_val,
    tracker=ConsoleTracker(k=1, regex=".*/loss"),
    train_batches=100,
    val_batches=10,
)

trainer.train(n_epochs=100)

for name, param in net.named_parameters():
    print(name, param.mean(), param.std())
