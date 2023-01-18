from itertools import islice

import torch
from torch import nn
from tqdm import tqdm


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
    ):
        self.net = net
        self.opt = make_opt(self.net.parameters())
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.tracker = tracker
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.loss_fn = nn.NLLLoss(reduction="sum")

    def train(self, n_epochs):
        for it in tqdm(range(n_epochs)):
            self.net.train()
            for x, y in islice(self.dl_train, self.train_batches):
                batch_size, seq_len = y.shape
                state = self.net.init_state(std=1, batch_size=batch_size)

                loss = 0
                for pos in range(seq_len):
                    p, state = self.net(x[:, pos], state)
                    loss += self.loss_fn(p, y[:, pos])
                loss /= batch_size

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.tracker.scalar("train/loss", loss)

            self.net.eval()
            with torch.no_grad():
                for x, y in islice(self.dl_val, self.val_batches):
                    batch_size, seq_len = y.shape
                    state = self.net.init_state(std=0, batch_size=batch_size)

                    loss = 0
                    for pos in range(seq_len):
                        p, state = self.net(x[:, pos], state)
                        loss += self.loss_fn(p, y[:, pos])
                    loss /= batch_size

                    self.tracker.scalar("val/loss", loss)
