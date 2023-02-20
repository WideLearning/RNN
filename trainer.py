from itertools import islice

import torch
from torch import nn
from tqdm import tqdm


class TrainerBase:
    def __init__(
        self,
        net,
        make_opt,
        dl_train,
        dl_val,
        tracker,
        train_batches=10**9,
        val_batches=10**9,
        l1=0.0,
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

    def _batch_loss(self, x, y):
        pass

    def validate(self):
        losses = []
        for x, y in islice(self.dl_val, self.val_batches):
            batch_size, _seq_len = y.shape

            loss = self._batch_loss(x, y) / batch_size
            losses.append(loss.item())
        return losses

    def train(self, n_epochs):
        for it in tqdm(range(n_epochs)):
            self.net.train()
            losses = []
            for x, y in islice(self.dl_train, self.train_batches):
                batch_size, _seq_len = y.shape

                loss = sum(w.abs().sum() for w in self.net.parameters()) * self.l1
                loss += self._batch_loss(x, y) / batch_size
                losses.append(loss.item())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            self.tracker.scalar("train/loss", sum(losses) / len(losses))
            self.tracker.model(self.net)
            self.net.eval()
            with torch.no_grad():
                losses = self.validate()
            self.tracker.scalar("val/loss", sum(losses) / len(losses))
        self.tracker.dump()


class RNNTrainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _batch_loss(self, x, y):
        batch_size, seq_len = y.shape
        loss = 0
        state = self.net.init_state(std=1, batch_size=batch_size)
        for pos in range(seq_len):
            p, state = self.net(x[:, pos], state)
            loss += self.loss_fn(p, y[:, pos])
        return loss


class FeedforwardTrainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _batch_loss(self, x, y):
        p = self.net(x).mT
        return self.loss_fn(p, y)
