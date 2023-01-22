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


activations_dict = {
    "identity": nn.Identity(),
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "logsoftmax": nn.LogSoftmax(dim=-1),
}

"""
Available extra configurations (for activation names see activations_dict):
hh_sg: True to cut gradients in recurrent connection, False by default
xh_sg: True to cut gradients in input connection, False by default
f_hh: activation for recurrent connection, tanh by default
f_hy: activation for output connection, logsoftmax by default
"""


class RNN(nn.Module):
    def __init__(self, x_size, h_size, y_size, config):
        super().__init__()
        self.h_size = h_size
        self.x_size = x_size
        self.y_size = y_size

        self.hh_sg = config.get("hh_sg", False)
        self.xh_sg = config.get("xh_sg", False)
        self.f_hh = activations_dict[config.get("f_hh", "tanh")]
        self.f_hy = activations_dict[config.get("f_hy", "logsoftmax")]

        self.W_hh = nn.Linear(self.h_size, h_size)
        self.W_xh = nn.Linear(self.x_size, self.h_size)
        self.W_hy = nn.Linear(self.h_size, self.y_size)

    def forward(self, x, state):
        state_linear = sg(self.W_hh(state), self.hh_sg)
        input_linear = sg(self.W_xh(x), self.xh_sg)
        new_state = self.f_hh(state_linear + input_linear)
        return self.f_hy(self.W_hy(new_state)), new_state

    def init_state(self, std=0, batch_size=1):
        return torch.randn((batch_size, self.h_size)) * std


"""
Available extra configurations:
hh_sg: True to cut gradients in recurrent connection, False by default
xh_sg: True to cut gradients in input connection, False by default
f_si: activation for input gate, sigmoid by default
f_sf: activation for forget gate, sigmoid by default
f_so: activation for output gate, sigmoid by default
f_su: activation for update, tanh by default
f_ch: activation for preoutput, tanh by default
f_hy: activation for output, logsoftmax by default
"""


class LSTM(nn.Module):
    def __init__(self, x_size, h_size, y_size, config):
        super().__init__()
        self.h_size = h_size
        self.x_size = x_size
        self.y_size = y_size

        self.hh_sg = config.get("hh_sg", False)
        self.xh_sg = config.get("xh_sg", False)
        self.f_si = activations_dict[config.get("f_si", "sigmoid")]
        self.f_sf = activations_dict[config.get("f_sf", "sigmoid")]
        self.f_so = activations_dict[config.get("f_so", "sigmoid")]
        self.f_su = activations_dict[config.get("f_su", "tanh")]
        self.f_ch = activations_dict[config.get("f_ch", "tanh")]
        self.f_hy = activations_dict[config.get("f_hy", "logsoftmax")]

        self.W_si = nn.Linear(self.x_size + self.h_size, self.h_size)
        self.W_oi = nn.Linear(self.x_size + self.h_size, self.h_size)
        self.W_fi = nn.Linear(self.x_size + self.h_size, self.h_size)
        self.W_ci = nn.Linear(self.x_size + self.h_size, self.h_size)
        self.W_hy = nn.Linear(self.h_size, self.y_size)

    def forward(self, x, state):
        h, c = state
        s = torch.cat((x, h))

        i_gate = self.f_si(self.W_si(s))
        o_gate = self.f_so(self.W_oi(s))
        f_gate = self.f_sf(self.W_fi(s))
        update = self.f_su(self.W_ci(s))
        new_c = f_gate * c + i_gate * update
        new_h = o_gate * self.f_o(new_c)
        return self.f_hy(self.W_hy(new_h)), (new_h, new_c)

    def init_state(self, std=0, batch_size=1):
        return (
            torch.randn((batch_size, self.h_size)) * std,
            torch.randn((batch_size, self.h_size)) * std,
        )
