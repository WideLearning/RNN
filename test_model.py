import unittest

import torch
from torch import nn

from rnn import RNN
from transformer import (
    LinearAttention,
    causal_attention_mask,
    linear_attn,
    softmax_attn,
)


class TestRNN(unittest.TestCase):
    def test_shapes(self):
        batch_size, h_size, x_size, y_size = 10, 4, 2, 3
        net = RNN(x_size=x_size, h_size=h_size, y_size=y_size, config={})
        state = net.init_state(std=1, batch_size=batch_size)
        data = torch.randn((batch_size, x_size))

        result, state = net(data, state)

        self.assertEqual(result.shape, (batch_size, y_size))
        self.assertEqual(state.shape, (batch_size, h_size))

    def test_growth(self):
        n_it, batch_size, h_size, x_size, y_size = 100, 10, 4, 2, 3
        net = RNN(h_size=h_size, x_size=x_size, y_size=y_size, config={})
        state = torch.randn((batch_size, h_size))
        loss = 0

        for i in range(n_it):
            data = torch.randn((batch_size, x_size))
            result, state = net(data, state)
            self.assertLess(result.abs().mean(), 10)
            loss += result.sum(dim=0)[0]

        loss.backward()
        for name, par in net.named_parameters():
            self.assertLess(par.grad.abs().mean() / n_it, 10)


class TestTransformer(unittest.TestCase):
    def test_lineat_attn(self):
        b, n, pos, d = 10, 7, 2, 16
        q = torch.randn((b, n, d), requires_grad=True)
        k = torch.randn((b, n, d), requires_grad=True)
        v = torch.randn((b, n, d), requires_grad=True)
        m = causal_attention_mask(n)

        y = linear_attn(k, v, q, m)
        y[:, pos].sum().backward()

        self.assertTrue(m.dtype is torch.float32)
        self.assertEqual(y.shape, (b, n, d))
        qg = q.grad.sum(dim=(0, 2))
        kg = k.grad.sum(dim=(0, 2))
        vg = v.grad.sum(dim=(0, 2))
        
        for i in range(n):
            self.assertEqual(qg[i] != 0, i == pos)
            self.assertEqual(kg[i] != 0, i < pos)
            self.assertEqual(vg[i] != 0, i < pos)

    def test_linear_attention(self):
        n, pos, d_x, d_h, d_y = 7, 2, 16, 10, 4
        x = torch.randn((n, d_x), requires_grad=True)
        a = LinearAttention(d_x, d_h, d_y)
        y = a(x)

        y[pos].sum().backward()

        self.assertEqual(y.shape, (n, d_y))
        xg = x.grad.sum(dim=1)
        for i in range(n):
            self.assertEqual(xg[i] != 0, i <= pos)
