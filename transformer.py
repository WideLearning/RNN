import torch
from torch import nn


def causal_attention_mask(n, device="cpu"):
    return torch.triu(torch.full((n, n), True, device=device), 1)


class LinearAttention(nn.Module):
    def __init__(self, x_size, h_size, y_size):
        super().__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.y_size = y_size
        self.W_Q = nn.Linear(self.x_size, self.h_size)
        self.W_K = nn.Linear(self.x_size, self.h_size)
        self.W_V = nn.Linear(self.x_size, self.y_size)
        self.A = nn.MultiheadAttention(
            kdim=self.h_size,
            vdim=self.y_size,
            batch_first=False,
        )

    def forward(self, x):
        n = x.size(0)
        assert x.shape == (n, self.x_size)
        query = self.W_Q(x)
        key = self.W_K(x)
        value = self.W_V(x)
        mask = causal_attention_mask(n, device=self.query.device)
        return self.A(query, key, value, need_weights=False)[0]
