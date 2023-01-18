import torch
import torch.nn as nn

"""
Based on https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html.

Note that the last layer is LogSoftmax, so use NLLLoss during training.

Takes input of shape (n_batch, x_size), returns output of shape (n_batch, y_size)
"""

# stop gradient operation, aka tensor.detach()
def sg(x, flag=True):
    if flag:
        return x.detach()
    else:
        return x


class RNN(nn.Module):
    def __init__(
        self,
        x_size,
        h_size,
        y_size,
        f_hh=nn.Tanh(),
        f_hy=nn.LogSoftmax(dim=1),
        hh_sg=False,
        xh_sg=False,
    ):
        super().__init__()
        self.h_size = h_size
        self.x_size = x_size
        self.y_size = y_size

        self.hh_sg = hh_sg
        self.xh_sg = hh_sg

        self.W_hh = nn.Linear(self.h_size, h_size)
        self.f_hh = f_hh
        self.W_xh = nn.Linear(self.x_size, self.h_size)
        self.W_hy = nn.Linear(self.h_size, self.y_size)
        self.f_hy = f_hy

    def forward(self, x, state):
        state_linear = sg(self.W_hh(state), self.hh_sg)
        input_linear = sg(self.W_xh(x), self.xh_sg)
        new_state = self.f_hh(state_linear + input_linear)
        return self.f_hy(self.W_hy(new_state)), new_state

    def init_state(self, std=0, batch_size=1):
        return torch.randn((batch_size, self.h_size)) * std
