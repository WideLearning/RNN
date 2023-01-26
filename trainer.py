from itertools import islice

import torch
from torch import nn
from tqdm import tqdm


def validate(net, dl_val, val_batches, loss_fn=nn.NLLLoss(reduction="sum")):
    with torch.no_grad():
        losses = []
        for x, y in islice(dl_val, val_batches):
            batch_size, seq_len = y.shape
            state = net.init_state(std=0, batch_size=batch_size)

            loss = 0
            for pos in range(seq_len):
                p, state = net(x[:, pos], state)
                loss += loss_fn(p, y[:, pos])
            loss /= batch_size
            losses.append(loss.item())
        return losses

class TeacherForcingTrainer:
    def __init__(
        self,
        net,
        make_opt,
        dl_train,
        dl_val,
        tracker,
        train_batches=10**9,
        val_batches=10**9,
        l1=0.0
    ):
        self.net = net
        self.tracker = tracker
        self.tracker.set_hooks(net)

        self.opt = make_opt(self.net.parameters())
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.loss_fn = nn.NLLLoss(reduction="sum")
        self.l1 = l1
    
    def train(self, n_epochs):
        for it in tqdm(range(n_epochs)):
            self.net.train()
            losses = []
            for x, y in islice(self.dl_train, self.train_batches):
                batch_size, seq_len = y.shape
                state = self.net.init_state(std=1, batch_size=batch_size)

                loss = sum(param.abs().sum() for param in self.net.parameters()) * self.l1
                for pos in range(seq_len):
                    p, state = self.net(x[:, pos], state)
                    loss += self.loss_fn(p, y[:, pos])
                loss /= batch_size
                losses.append(loss.item())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            self.tracker.scalar("train/loss", sum(losses) / len(losses))
            self.tracker.model(self.net)
            self.net.eval()
            with torch.no_grad():
                losses = validate(self.net, self.dl_val, self.val_batches)
            self.tracker.scalar("val/loss", sum(losses) / len(losses))
        self.tracker.dump()
