import torch

from data import ptb_loaders
from model import RNN
from TeleBoard.tracker import ConsoleTracker
from trainer import TeacherForcingTrainer

dl_train, dl_val = ptb_loaders(
    n_seq=10,
    n_batch=512,
)
net = RNN(
    n_x=128,
    n_h=256,
    n_y=128,
)
trainer = TeacherForcingTrainer(
    net=net,
    make_opt=lambda p: torch.optim.Adam(p),
    dl_train=dl_train,
    dl_val=dl_val,
    tracker=ConsoleTracker(k=1, regex=".*/loss"),
)

trainer.train(n_epochs=3)
