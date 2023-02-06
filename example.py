import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import ptb_loaders
from rnn import RNN
from TeleBoard.tracker import FileTracker
from trainer import TeacherForcingTrainer

dl_train, dl_val = ptb_loaders(
    seq_len=5,
    batch_size=128,
)
net = RNN(
    x_size=128,
    h_size=1024,
    y_size=128,
)
trainer = TeacherForcingTrainer(
    net=net,
    make_opt=lambda p: torch.optim.Adam(p),
    dl_train=dl_train,
    dl_val=dl_val,
    tracker=FileTracker(k=5, filename="save.p"),
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
        state = net.init_state(std=0,batch_size=1)
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
