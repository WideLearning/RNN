import torch
from torch import nn
from tqdm import tqdm


class TeacherForcingTrainer:
    def __init__(self, net, make_opt, dl_train, dl_val, logger):
        self.net = net
        self.opt = make_opt(self.net.parameters())
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.logger = logger
        self.loss_fn = nn.NLLLoss()

    def train(self, n_epochs):
        for it in tqdm(range(n_epochs)):
            self.net.train()
            for x, y in self.dl_train:
                assert x.shape == y.shape
                n_seq, n_batch, _ = x.shape
                state = self.net.init_state(std=1, n_b=n_batch)

                loss = 0
                for pos in range(n_seq):
                    p, state = self.net(x[pos], state)
                    loss += self.loss_fn(p, y[pos])
                loss /= n_seq

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.logger.scalar("train/loss", loss)

            self.net.eval()
            with torch.no_grad():
                for x, y in self.dl_val:
                    n_seq, n_batch, _ = x.shape
                    state = self.net.init_state(std=0, n_b=n_batch)

                    loss = 0
                    for pos in range(n_seq):
                        p, state = self.net(x[pos], state)
                        loss += self.loss_fn(p, y[pos])
                    loss /= n_seq

                    self.logger.scalar("val/loss", loss)
