from itertools import islice

import torch
from torch import nn
from tqdm import tqdm


class TeacherForcingTrainer:
    def __init__(self, net, make_opt, dl_train, dl_val, tracker, train_batches=10**9, val_batches=10**9):
        self.net = net
        self.opt = make_opt(self.net.parameters())
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.tracker = tracker
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.loss_fn = nn.NLLLoss()

    def train(self, n_epochs):
        for it in tqdm(range(n_epochs)):
            self.net.train()
            for x, y in islice(self.dl_train, self.train_batches):
                n_batch, n_seq = y.shape
                state = self.net.init_state(std=1, n_batch=n_batch)

                loss = 0
                for pos in range(n_seq):
                    p, state = self.net(x[:, pos], state)
                    loss += self.loss_fn(p, y[:, pos])
                loss /= n_seq

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.tracker.scalar("train/loss", loss)

            self.net.eval()
            with torch.no_grad():
                for x, y in islice(self.dl_val, self.val_batches):
                    n_batch, n_seq = y.shape
                    state = self.net.init_state(std=0, n_batch=n_batch)

                    loss = 0
                    for pos in range(n_seq):
                        p, state = self.net(x[:, pos], state)
                        loss += self.loss_fn(p, y[:, pos])
                    loss /= n_seq

                    self.tracker.scalar("val/loss", loss)
