import torch
from torch import nn


def causal_attention_mask(n: int, device="cpu") -> torch.Tensor:
    """
    >>> causal_attention_mask(3)
    tensor([[0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.]])
    """
    return torch.tril(torch.ones((n, n), device=device), -1)


def _attn(
    K: torch.Tensor,
    V: torch.Tensor,
    Q: torch.Tensor,
    mask=None,
):
    n, d_v = V.shape[-2:]
    m, d_kq = Q.shape[-2:]
    if mask is None:
        mask = torch.ones((n, m), device=V.device)
    assert V.shape[-2:] == (n, d_v)
    assert K.shape[-2:] == (n, d_kq)
    assert Q.shape[-2:] == (m, d_kq)
    assert mask.shape == (m, n)

    W = (Q @ K.mT) * mask
    return W, V, d_kq


def linear_attn(
    K: torch.Tensor,
    V: torch.Tensor,
    Q: torch.Tensor,
    mask=None,
) -> torch.Tensor:
    """
    Dot product attention without softmax and normalization.

    `linear_attn(K, V, Q) = (Q K^{T} + mask) V`

    where:

    `V`: float tensor, (..., n, d_{v})
        Value vectors.
    `K`: float tensor, (..., n, d_{kq})
        Key vectors.
    `Q`: float tensor, (..., m, d_{kq})
        Query vectors.
    `mask`: float tensor, (m, n)
        `mask[i][j]` = whether `i`-th query should attend to `j`-th value.
        Using ones(n, m) if set to None.
    returns: float tensor, (m, d_{v})
    """
    W, V, _d_kq = _attn(K, V, Q, mask)
    return W @ V


def softmax_attn(
    K: torch.Tensor,
    V: torch.Tensor,
    Q: torch.Tensor,
    mask=None,
) -> torch.Tensor:
    """
    Dot product attention with softmax and normalization.

    `softmax_attn(K, V, Q) = softmax(Q K^{T} / sqrt(d_{kq})) V`

    where:

    `V`: float tensor, (..., n, d_{v})
        Value vectors.
    `K`: float tensor, (..., n, d_{kq})
        Key vectors.
    `Q`: float tensor, (..., m, d_{kq})
        Query vectors.
    `mask`: float tensor, (m, n)
        `mask[i][j]` = whether `i`-th query should attend to `j`-th value.
        Using ones(n, m) if set to None.
    returns: float tensor, (m, d_{v})
    """
    W, V, d_kq = _attn(K, V, Q, mask)
    return torch.softmax(W / (d_kq**0.5), dim=-2) @ V


class LinearAttention(nn.Module):
    """
    Applies embeddings to get key, value and query from input, then computes dot product attention.

    `LinearAttention(x) = linear_attn(W_{K} x, W_{V} x, W_{Q} x)`
    """

    def __init__(self, x_size: int, h_size: int, y_size: int):
        """
        `x_size`: int
            Embedding dimension of input.
        `h_size`: int
            Embedding dimension of key and query vectors.
        `y_size`: int
            Embedding dimension of value vectors.
        """
        super().__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.y_size = y_size
        self.W_K = nn.Linear(self.x_size, self.h_size)
        self.W_V = nn.Linear(self.x_size, self.y_size)
        self.W_Q = nn.Linear(self.x_size, self.h_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        `x`: float tensor, (..., n, x_size)

        returns: float tensor, (..., n, y_size)
        """
        n = x.size(-2)
        assert x.shape[-2:] == (n, self.x_size)
        query = self.W_Q(x)
        key = self.W_K(x)
        value = self.W_V(x)
        mask = causal_attention_mask(n, device=value.device)
        result = linear_attn(key, value, query, mask)
        assert result.shape[-2:] == (n, self.y_size)
        return result


class SoftmaxAttention(nn.Module):
    """
    Applies embeddings to get key, value and query from input, then computes normalized softmax dot product attention.

    `SoftmaxAttention(x) = softmax_attn(W_{K} x, W_{V} x, W_{Q} x)`
    """

    def __init__(self, x_size: int, h_size: int, y_size: int):
        """
        `x_size`: int
            Embedding dimension of input.
        `h_size`: int
            Embedding dimension of key and query vectors.
        `y_size`: int
            Embedding dimension of value vectors.
        """
        super().__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.y_size = y_size
        self.W_K = nn.Linear(self.x_size, self.h_size)
        self.W_V = nn.Linear(self.x_size, self.y_size)
        self.W_Q = nn.Linear(self.x_size, self.h_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        `x`: float tensor, (..., n, x_size)

        returns: float tensor, (..., n, y_size)
        """
        n = x.size(-2)
        assert x.shape[-2:] == (n, self.x_size)
        query = self.W_Q(x)
        key = self.W_K(x)
        value = self.W_V(x)
        mask = causal_attention_mask(n, device=value.device)
        result = softmax_attn(key, value, query, mask)
        assert result.shape[-2:] == (n, self.y_size)
        return result


class SumParallel(nn.Module):
    """
    SumParallel(x) = f(x) + g(x)
    """

    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x):
        return self.f(x) + self.g(x)