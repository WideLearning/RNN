import torch
import torch.nn as nn


"""
Based on https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html.

Note that the last layer is LogSoftmax, so use NLLLoss during training.

Takes input of shape (n_batch, n_x), returns output of shape (n_batch, n_y)
"""


class RNN(nn.Module):
    def __init__(self, n_x, n_h, n_y):
        super().__init__()
        self.n_h = n_h
        self.n_x = n_x
        self.n_y = n_y

        self.W_hh = nn.Linear(self.n_h, n_h)
        self.f_hh = nn.Tanh()
        self.W_xh = nn.Linear(self.n_x, self.n_h)
        self.W_hy = nn.Linear(self.n_h, self.n_y)
        self.f_hy = nn.LogSoftmax(dim=1)

    def forward(self, x, state):
        new_state = self.f_hh(self.W_hh(state) + self.W_xh(x))
        return self.f_hy(self.W_hy(new_state)), new_state

    def init_state(self, std=0, n_batch=1):
        return torch.randn((n_batch, self.n_h)) * std
