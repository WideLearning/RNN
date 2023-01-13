import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import ptb_loaders
from model import RNN
from TeleBoard.tracker import ConsoleTracker
from trainer import TeacherForcingTrainer

dl_train, dl_val = ptb_loaders(
    n_seq=10,
    n_batch=1024,
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
    train_batches=100,
    val_batches=10
)


def to_onehot(c):
    return F.one_hot(torch.tensor([c]), num_classes=128).float()


for epoch in range(100):
    trainer.train(n_epochs=1)
    torch.save(net.state_dict(), "rnn.tmp")

    with torch.no_grad():
        print(f"Epoch #{epoch}")
        prompt = "The quick brown"
        onehot = [to_onehot(ord(c)) for c in prompt]
        state = net.init_state(std=0, n_batch=1)
        logprobs = None
        for x in onehot:
            logprobs, state = net(x, state)
        for it in range(100):
            probs = logprobs.exp().numpy().ravel()
            c = np.random.choice(128, p=probs)
            prompt += chr(c)
            onehot.append(to_onehot(c))
            logprobs, state = net(x, state)
        print(prompt)
